from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_ipcw
import torchtuples as tt
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_dataset import (  # noqa: E402
    DATE_COL,
    DURATION_COL,
    EVENT_COL,
    TECHNICAL_FEATURE_COLS,
    load_gold_dataset,
    prepare_dataset,
    select_technical_feature_columns,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) <= 1:
        return float("nan")
    return float(roc_auc_score(y, p))


def _top_decile(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(prob) == 0:
        return float("nan"), float("nan")
    k = max(1, int(len(prob) * 0.1))
    idx = np.argsort(prob)[-k:]
    top = float(np.mean(y[idx]))
    base = float(np.mean(y))
    lift = top / base if base > 0 else float("nan")
    return top, lift


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (p >= left) & (p < right if i < bins - 1 else p <= right)
        if not np.any(mask):
            continue
        ece += abs(float(np.mean(y[mask])) - float(np.mean(p[mask]))) * (np.sum(mask) / len(y))
    return float(ece)


def _macro_f1_3class(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in [0, 1, 2]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return float(np.mean(f1s))


def _confusion_3class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _to_surv_struct(duration: np.ndarray, event_binary: np.ndarray) -> np.ndarray:
    return np.array(
        list(zip(event_binary.astype(bool), duration.astype(float), strict=False)),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def uno_c_index(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    risk_score: np.ndarray,
    cause: int,
    horizon: int,
) -> float:
    train_duration = train_frame[DURATION_COL].values.astype(float)
    eval_duration = eval_frame[DURATION_COL].values.astype(float)

    train_event = (train_frame[EVENT_COL].values.astype(int) == cause).astype(int)
    eval_event = (eval_frame[EVENT_COL].values.astype(int) == cause).astype(int)

    y_train = _to_surv_struct(train_duration, train_event)
    y_eval = _to_surv_struct(eval_duration, eval_event)

    tau = float(horizon)
    cidx, *_ = concordance_index_ipcw(y_train, y_eval, risk_score.astype(float), tau=tau)
    return float(cidx)


def _surv_matrix_at_times(surv_df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    times_f = times.astype(float)
    idx = surv_df.index.to_numpy(dtype=float)
    union = np.unique(np.concatenate([idx, times_f]))
    aligned = surv_df.reindex(union).sort_index().ffill().bfill()
    surv = aligned.loc[times_f].to_numpy()
    return np.clip(surv, 1e-12, 1.0)


def _fit_cause_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    event_col: str,
    num_nodes: list[int],
    batch_norm: bool,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int,
    verbose: bool,
) -> CoxTime:
    x_train = train[feature_cols].to_numpy(dtype=np.float32)
    x_valid = valid[feature_cols].to_numpy(dtype=np.float32)

    durations_train = train[DURATION_COL].values.astype("float32")
    durations_valid = valid[DURATION_COL].values.astype("float32")
    events_train = train[event_col].values.astype("int64")
    events_valid = valid[event_col].values.astype("int64")

    labtrans = CoxTime.label_transform()
    y_train = labtrans.fit_transform(durations_train, events_train)
    y_valid = labtrans.transform(durations_valid, events_valid)

    net = MLPVanillaCoxTime(
        in_features=len(feature_cols),
        num_nodes=num_nodes,
        batch_norm=batch_norm,
        dropout=dropout,
    )
    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
    model.optimizer.set_lr(lr)

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        val_data=(x_valid, y_valid),
        verbose=verbose,
    )
    model.compute_baseline_hazards()
    return model


def _predict_cause_surv(
    model: CoxTime,
    frame: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = frame[feature_cols].to_numpy(dtype=np.float32)
    surv_df = model.predict_surv_df(x)

    times = np.arange(1, horizon + 1)
    surv_mat = _surv_matrix_at_times(surv_df, times)  # [t, n]
    ch_mat = -np.log(np.clip(surv_mat, 1e-12, 1.0))

    p_by_h = np.clip(1.0 - surv_mat[-1, :], 0.0, 1.0)
    return ch_mat, p_by_h


def _competing_cif(
    up_ch: np.ndarray,
    down_ch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_ch = up_ch + down_ch
    up_inc = np.diff(np.vstack([np.zeros((1, up_ch.shape[1])), up_ch]), axis=0)
    down_inc = np.diff(np.vstack([np.zeros((1, down_ch.shape[1])), down_ch]), axis=0)
    total_prev = np.vstack([np.zeros((1, total_ch.shape[1])), total_ch[:-1, :]])
    surv_prev = np.exp(-total_prev)

    cif_up = np.cumsum(surv_prev * up_inc, axis=0)
    cif_down = np.cumsum(surv_prev * down_inc, axis=0)
    return np.clip(cif_up, 0.0, 1.0), np.clip(cif_down, 0.0, 1.0)


def _split_metrics(
    frame: pd.DataFrame,
    p_up_h: np.ndarray,
    p_down_h: np.ndarray,
    cif_up: np.ndarray,
    cif_down: np.ndarray,
    train_frame: pd.DataFrame,
    horizon: int,
    td_times: list[int],
    split_name: str,
    use_tqdm: bool,
    post_pbar: tqdm | None = None,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, pd.DataFrame]:
    y_up = (frame[EVENT_COL].values == 1).astype(int)
    y_down = (frame[EVENT_COL].values == 2).astype(int)
    y_3 = frame[EVENT_COL].values.astype(int)

    p_no_h = np.clip(1.0 - p_up_h - p_down_h, 0.0, 1.0)
    pred_3 = np.argmax(np.column_stack([p_no_h, p_up_h, p_down_h]), axis=1)

    up_top, up_lift = _top_decile(p_up_h, y_up)
    down_top, down_lift = _top_decile(p_down_h, y_down)
    if post_pbar is not None:
        post_pbar.set_postfix(step=f"{split_name}:base")
    uno_up = uno_c_index(train_frame, frame, p_up_h, cause=1, horizon=horizon)
    if post_pbar is not None:
        post_pbar.update(1)
        post_pbar.set_postfix(step=f"{split_name}:uno_up", uno_up=f"{uno_up:.3f}")
    uno_down = uno_c_index(train_frame, frame, p_down_h, cause=2, horizon=horizon)
    if post_pbar is not None:
        post_pbar.update(1)
        post_pbar.set_postfix(step=f"{split_name}:uno_down", uno_down=f"{uno_down:.3f}")

    metrics = {
        "n": int(len(frame)),
        "event_rate_up": float(np.mean(y_up)),
        "event_rate_down": float(np.mean(y_down)),
        "mean_p_up_30d": float(np.mean(p_up_h)),
        "mean_p_down_30d": float(np.mean(p_down_h)),
        "mean_p_no_hit_30d": float(np.mean(p_no_h)),
        "sum_prob_le_1_rate": float(np.mean((p_up_h + p_down_h) <= 1.000001)),
        "brier_up": float(np.mean((p_up_h - y_up) ** 2)),
        "brier_down": float(np.mean((p_down_h - y_down) ** 2)),
        "brier_overall": float(np.mean((p_no_h - (y_3 == 0).astype(int)) ** 2 + (p_up_h - y_up) ** 2 + (p_down_h - y_down) ** 2)),
        "ece_up": _ece(y_up, p_up_h, bins=10),
        "ece_down": _ece(y_down, p_down_h, bins=10),
        "top_decile_hit_up": up_top,
        "top_decile_lift_up": up_lift,
        "top_decile_hit_down": down_top,
        "top_decile_lift_down": down_lift,
        "td_auc_up_30d": _safe_auc(y_up, p_up_h),
        "td_auc_down_30d": _safe_auc(y_down, p_down_h),
        "uno_c_up": uno_up,
        "uno_c_down": uno_down,
        "acc_3class": float(np.mean(pred_3 == y_3)),
        "macro_f1_3class": _macro_f1_3class(y_3, pred_3),
    }

    duration = frame[DURATION_COL].values.astype(int)
    event = frame[EVENT_COL].values.astype(int)
    rows: list[dict[str, float | int]] = []
    td_iter = tqdm(td_times, desc=f"TD {split_name}", dynamic_ncols=True, leave=False, disable=not use_tqdm)
    for t in td_iter:
        y_up_t = ((event == 1) & (duration <= t)).astype(int)
        y_down_t = ((event == 2) & (duration <= t)).astype(int)
        p_up_t = cif_up[t - 1, :]
        p_down_t = cif_down[t - 1, :]
        p_no_t = np.clip(1.0 - p_up_t - p_down_t, 0.0, 1.0)

        y_cls_t = np.where((event == 1) & (duration <= t), 1, np.where((event == 2) & (duration <= t), 2, 0))
        y_onehot = np.eye(3)[y_cls_t]
        p_mat = np.column_stack([p_no_t, p_up_t, p_down_t])
        brier_overall_t = float(np.mean(np.sum((p_mat - y_onehot) ** 2, axis=1)))

        rows.append(
            {
                "t": t,
                "auc_up": _safe_auc(y_up_t, p_up_t),
                "auc_down": _safe_auc(y_down_t, p_down_t),
                "brier_up": float(np.mean((p_up_t - y_up_t) ** 2)),
                "brier_down": float(np.mean((p_down_t - y_down_t) ** 2)),
                "brier_overall": brier_overall_t,
            }
        )

    td_df = pd.DataFrame(rows)
    metrics.update(
        {
            "ibs_up": float(td_df["brier_up"].mean()),
            "ibs_down": float(td_df["brier_down"].mean()),
            "ibs_overall": float(td_df["brier_overall"].mean()),
            "td_auc_up_mean": float(td_df["auc_up"].dropna().mean()) if td_df["auc_up"].notna().any() else float("nan"),
            "td_auc_down_mean": float(td_df["auc_down"].dropna().mean()) if td_df["auc_down"].notna().any() else float("nan"),
        }
    )

    pred_df = frame[["ticker", DATE_COL, DURATION_COL, EVENT_COL]].copy()
    pred_df["p_up_30d"] = p_up_h
    pred_df["p_down_30d"] = p_down_h
    pred_df["p_no_hit_30d"] = p_no_h
    pred_df["pred_3class"] = pred_3

    return metrics, pred_df, _confusion_3class(y_3, pred_3), td_df


def _plot_calibration(pred_df: pd.DataFrame, cause: str, outdir: Path) -> Path:
    y = (pred_df[EVENT_COL].values == (1 if cause == "up" else 2)).astype(int)
    p = pred_df[f"p_{cause}_30d"].values
    bins = np.linspace(0, 1, 11)
    mids, obs = [], []

    for i in range(10):
        l, r = bins[i], bins[i + 1]
        m = (p >= l) & (p < r if i < 9 else p <= r)
        if not np.any(m):
            continue
        mids.append(float(np.mean(p[m])))
        obs.append(float(np.mean(y[m])))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.plot(mids, obs, marker="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"Calibration @30d ({cause})")
    fig.tight_layout()

    path = outdir / f"calibration_{cause}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_cif(cif_up: np.ndarray, cif_down: np.ndarray, outdir: Path) -> Path:
    t = np.arange(1, cif_up.shape[0] + 1)
    mean_up = cif_up.mean(axis=1)
    mean_down = cif_down.mean(axis=1)
    no_hit = np.clip(1 - mean_up - mean_down, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean_up, label="CIF up")
    ax.plot(t, mean_down, label="CIF down")
    ax.plot(t, no_hit, label="No-hit")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day")
    ax.set_ylabel("Probability")
    ax.set_title("Mean CIF curves (test)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()

    path = outdir / "cif_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_td(td_df: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(td_df["t"], td_df["auc_up"], label="AUC up")
    axes[0].plot(td_df["t"], td_df["auc_down"], label="AUC down")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Time-dependent AUC")
    axes[0].set_xlabel("t (day)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].plot(td_df["t"], td_df["brier_up"], label="Brier up")
    axes[1].plot(td_df["t"], td_df["brier_down"], label="Brier down")
    axes[1].plot(td_df["t"], td_df["brier_overall"], label="Brier overall")
    axes[1].set_title("Brier by time")
    axes[1].set_xlabel("t (day)")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")
    fig.tight_layout()

    path = outdir / "time_dependent_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cause-specific CoxTime baseline (2-model competing risks)")
    p.add_argument("--data-path", type=Path, default=Path("data/gold/survival_dataset.parquet"))
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--valid-ratio", type=float, default=0.15)
    p.add_argument("--train-end", type=str, default=None)
    p.add_argument("--valid-end", type=str, default=None)
    p.add_argument("--min-date", type=str, default=None)
    p.add_argument("--max-date", type=str, default=None)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--scale-method", type=str, default="train_standard", choices=["train_standard", "none"])
    p.add_argument("--winsor-lower-q", type=float, default=0.005)
    p.add_argument("--winsor-upper-q", type=float, default=0.995)
    p.add_argument("--extreme-abs-threshold", type=float, default=1e4)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-norm", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-splits", type=str, default="test", help="Comma-separated: train,valid,test")
    p.add_argument("--td-times", type=str, default="5,10,20,30", help="Comma-separated times for td-AUC/Brier")
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--outdir", type=Path, default=Path("experiments/deep/outputs/coxtime"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    use_tqdm = not args.no_tqdm
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    allowed = {"train", "valid", "test"}
    if not eval_splits or any(s not in allowed for s in eval_splits):
        raise ValueError("--eval-splits must be comma-separated subset of: train,valid,test")
    td_times = sorted({int(x.strip()) for x in args.td_times.split(",") if x.strip()})
    td_times = [t for t in td_times if 1 <= t <= args.horizon]
    if not td_times:
        raise ValueError("--td-times must include at least one integer within [1, horizon]")

    set_seed(args.seed)

    df = load_gold_dataset(args.data_path, min_date=args.min_date, max_date=args.max_date)
    feature_cols = select_technical_feature_columns(df)
    missing = sorted(set(TECHNICAL_FEATURE_COLS) - set(feature_cols))
    if missing:
        print(f"[warn] missing features: {missing}")
    if not feature_cols:
        raise ValueError("No requested technical features found.")

    splits = prepare_dataset(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        train_end=args.train_end,
        valid_end=args.valid_end,
        min_date=args.min_date,
        max_date=args.max_date,
        feature_cols=feature_cols,
        scale_method=args.scale_method,
        winsor_lower_q=args.winsor_lower_q,
        winsor_upper_q=args.winsor_upper_q,
        extreme_abs_threshold=args.extreme_abs_threshold,
    )

    print(
        f"[split] train={len(splits.train):,} valid={len(splits.valid):,} test={len(splits.test):,} "
        f"features={len(splits.feature_cols)}"
    )

    num_nodes = [args.hidden_dim, args.hidden_dim]

    t0 = time.perf_counter()
    print("[fit] training CoxTime up")
    up_model = _fit_cause_model(
        splits.train,
        splits.valid,
        splits.feature_cols,
        event_col="event_up",
        num_nodes=num_nodes,
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )
    print("[fit] training CoxTime down")
    down_model = _fit_cause_model(
        splits.train,
        splits.valid,
        splits.feature_cols,
        event_col="event_down",
        num_nodes=num_nodes,
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )
    print(f"[time] fit={time.perf_counter() - t0:.1f}s")

    split_map = {"train": splits.train, "valid": splits.valid, "test": splits.test}
    split_items = [(name, split_map[name]) for name in eval_splits]
    post_pbar = tqdm(
        total=len(split_items) * 5,
        desc="Postprocess",
        dynamic_ncols=True,
        disable=not use_tqdm,
    )

    metrics: dict[str, dict[str, float]] = {}
    preds: dict[str, pd.DataFrame] = {}
    cms: dict[str, np.ndarray] = {}
    td_frames: dict[str, pd.DataFrame] = {}
    test_cif: tuple[np.ndarray, np.ndarray] | None = None

    t1 = time.perf_counter()
    for split_name, frame in split_items:
        if use_tqdm:
            post_pbar.set_postfix(step=f"{split_name}:predict_up")
        up_ch, p_up_h = _predict_cause_surv(up_model, frame, splits.feature_cols, args.horizon)
        if use_tqdm:
            post_pbar.update(1)
            post_pbar.set_postfix(step=f"{split_name}:predict_down")
        down_ch, p_down_h = _predict_cause_surv(down_model, frame, splits.feature_cols, args.horizon)
        if use_tqdm:
            post_pbar.update(1)
        cif_up, cif_down = _competing_cif(up_ch, down_ch)

        if split_name == "test":
            test_cif = (cif_up, cif_down)

        m, p, cm, td_df = _split_metrics(
            frame,
            p_up_h,
            p_down_h,
            cif_up,
            cif_down,
            splits.train,
            horizon=args.horizon,
            td_times=td_times,
            split_name=split_name,
            use_tqdm=use_tqdm,
            post_pbar=post_pbar if use_tqdm else None,
        )
        metrics[split_name] = m
        preds[split_name] = p
        cms[split_name] = cm
        td_frames[split_name] = td_df

        if use_tqdm:
            post_pbar.update(1)
            post_pbar.set_postfix(step=f"{split_name}:td_done", uno_up=f"{m['uno_c_up']:.3f}", uno_down=f"{m['uno_c_down']:.3f}")

    print(f"[time] predict_eval={time.perf_counter() - t1:.1f}s")
    if use_tqdm:
        post_pbar.close()

    if "test" in preds:
        if test_cif is None:
            raise RuntimeError("test split CIF missing")
        _plot_calibration(preds["test"], "up", args.outdir)
        _plot_calibration(preds["test"], "down", args.outdir)
        _plot_cif(test_cif[0], test_cif[1], args.outdir)
        _plot_td(td_frames["test"], args.outdir)
        preds["test"].to_parquet(args.outdir / "test_predictions.parquet", index=False)
        pd.DataFrame(
            cms["test"],
            index=["true_no_hit", "true_up", "true_down"],
            columns=["pred_no_hit", "pred_up", "pred_down"],
        ).to_csv(args.outdir / "confusion_3class_test.csv")

    if "test" in td_frames:
        td_frames["test"].to_csv(args.outdir / "time_dependent_metrics_test.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(args.outdir / "metrics.csv", index=True)

    summary = {
        "config": {
            "data_path": str(args.data_path),
            "horizon": args.horizon,
            "features_used": splits.feature_cols,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_norm": args.batch_norm,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "eval_splits": eval_splits,
            "td_times": td_times,
            "winsor_lower_q": args.winsor_lower_q,
            "winsor_upper_q": args.winsor_upper_q,
            "extreme_abs_threshold": args.extreme_abs_threshold,
            "train_end": str(splits.train_end),
            "valid_end": str(splits.valid_end),
        },
        "data_quality": splits.quality_report,
        "metrics": metrics,
        "notes": [
            "CoxTime is fit separately for up/down causes (cause-specific hazards).",
            "30d CIF is derived by combining two cause-specific cumulative hazards.",
        ],
    }
    (args.outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] outdir={args.outdir}")
    report_key = "test" if "test" in metrics else eval_splits[0]
    print(
        f"[done] {report_key} metrics: acc_3class={metrics[report_key]['acc_3class']:.4f}, "
        f"macro_f1_3class={metrics[report_key]['macro_f1_3class']:.4f}, "
        f"uno_up={metrics[report_key]['uno_c_up']:.4f}, uno_down={metrics[report_key]['uno_c_down']:.4f}, "
        f"ibs_overall={metrics[report_key]['ibs_overall']:.4f}"
    )


if __name__ == "__main__":
    main()
