from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_dataset import (
    DATE_COL,
    DURATION_COL,
    EVENT_COL,
    TECHNICAL_FEATURE_COLS,
    load_gold_dataset,
    prepare_dataset,
    select_technical_feature_columns,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install in research env: conda install -n research -c conda-forge matplotlib"
    ) from exc

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from lifelines.utils import concordance_index
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "lifelines is required. Install in research env: conda install -n research -c conda-forge lifelines"
    ) from exc


def _safe_concordance(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    if np.unique(event).size <= 1:
        return float("nan")
    return float(concordance_index(duration, -risk, event))


def _top_decile_stats(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(prob) == 0:
        return float("nan"), float("nan")
    k = max(1, int(len(prob) * 0.1))
    idx = np.argsort(prob)[-k:]
    top_rate = float(np.mean(y[idx]))
    base_rate = float(np.mean(y))
    lift = top_rate / base_rate if base_rate > 0 else float("nan")
    return top_rate, lift


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (p >= left) & (p < right if i < bins - 1 else p <= right)
        if not np.any(mask):
            continue
        obs = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece += abs(obs - conf) * (np.sum(mask) / len(y))
    return float(ece)


def _predict_cause(
    model: CoxPHFitter,
    frame: pd.DataFrame,
    feature_cols: list[str],
    event_col: str,
    cause: str,
    horizon: int,
) -> pd.DataFrame:
    x = frame[feature_cols]
    risk = model.predict_partial_hazard(x).values.ravel()
    ch = model.predict_cumulative_hazard(x, times=[horizon])
    haz_h = ch.iloc[-1].values
    prob_h = 1.0 - np.exp(-haz_h)

    return pd.DataFrame(
        {
            "ticker": frame["ticker"].reset_index(drop=True),
            DATE_COL: frame[DATE_COL].reset_index(drop=True),
            DURATION_COL: frame[DURATION_COL].reset_index(drop=True),
            EVENT_COL: frame[EVENT_COL].reset_index(drop=True),
            event_col: frame[event_col].reset_index(drop=True),
            f"risk_{cause}": risk,
            f"p_{cause}_by_{horizon}d": np.clip(prob_h, 0.0, 1.0),
        }
    )


def _metrics_for_split(pred: pd.DataFrame, event_col: str, risk_col: str, prob_col: str) -> dict[str, float]:
    y = pred[event_col].values.astype(int)
    risk = pred[risk_col].values
    prob = pred[prob_col].values

    cidx = _safe_concordance(pred[DURATION_COL].values, y, risk)
    top_rate, top_lift = _top_decile_stats(prob, y)

    return {
        "n": int(len(pred)),
        "event_rate": float(np.mean(y)),
        "mean_pred_prob": float(np.mean(prob)),
        "c_index": cidx,
        "brier": float(np.mean((prob - y) ** 2)),
        "log_loss": _log_loss(y, prob),
        "ece_10bin": _ece(y, prob, bins=10),
        "top_decile_hit_rate": top_rate,
        "top_decile_lift": top_lift,
    }


def _plot_ph_violations(ph_df: pd.DataFrame, cause: str, outdir: Path, alpha: float) -> Path:
    top = ph_df.sort_values("p_value", ascending=True).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["minus_log10_p"], color="#3A7CA5")
    ax.axvline(-np.log10(alpha), color="#D1495B", linestyle="--", label=f"alpha={alpha}")
    ax.set_title(f"PH violations ({cause})")
    ax.set_xlabel("-log10(p-value)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = outdir / f"ph_violations_{cause}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_calibration(pred: pd.DataFrame, event_col: str, prob_col: str, cause: str, outdir: Path) -> Path:
    y = pred[event_col].values.astype(int)
    p = pred[prob_col].values
    bins = np.linspace(0, 1, 11)
    mids, obs = [], []

    for i in range(10):
        left, right = bins[i], bins[i + 1]
        mask = (p >= left) & (p < right if i < 9 else p <= right)
        if not np.any(mask):
            continue
        mids.append(float(np.mean(p[mask])))
        obs.append(float(np.mean(y[mask])))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.plot(mids, obs, marker="o", color="#2A9D8F")
    ax.set_title(f"Calibration @30d ({cause})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    path = outdir / f"calibration_{cause}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_risk_hist(pred: pd.DataFrame, event_col: str, risk_col: str, cause: str, outdir: Path) -> Path:
    y = pred[event_col].values.astype(int)
    risk = pred[risk_col].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(risk[y == 0], bins=40, alpha=0.6, label="non-event", color="#577590", density=True)
    ax.hist(risk[y == 1], bins=40, alpha=0.6, label="event", color="#F3722C", density=True)
    ax.set_title(f"Risk score distribution ({cause})")
    ax.set_xlabel("Partial hazard")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    fig.tight_layout()

    path = outdir / f"risk_hist_{cause}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_hr(model: CoxPHFitter, cause: str, outdir: Path) -> Path:
    s = model.summary[["coef"]].copy()
    s["hr"] = np.exp(s["coef"])
    s["log_hr"] = np.log(s["hr"])
    top = s.reindex(s["log_hr"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top["log_hr"], color="#90BE6D")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title(f"Top |log(HR)| ({cause})")
    ax.set_xlabel("log(HR)")
    fig.tight_layout()

    path = outdir / f"log_hr_{cause}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _compute_cif_curves(
    up_model: CoxPHFitter,
    down_model: CoxPHFitter,
    test: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x = test[feature_cols]
    times = np.arange(1, horizon + 1)

    up_ch = up_model.predict_cumulative_hazard(x, times=times).to_numpy()
    down_ch = down_model.predict_cumulative_hazard(x, times=times).to_numpy()

    total_ch = up_ch + down_ch
    up_inc = np.diff(np.vstack([np.zeros((1, up_ch.shape[1])), up_ch]), axis=0)
    down_inc = np.diff(np.vstack([np.zeros((1, down_ch.shape[1])), down_ch]), axis=0)
    total_prev = np.vstack([np.zeros((1, total_ch.shape[1])), total_ch[:-1, :]])
    surv_prev = np.exp(-total_prev)

    cif_up = np.cumsum(surv_prev * up_inc, axis=0)
    cif_down = np.cumsum(surv_prev * down_inc, axis=0)

    out = test[["ticker", DATE_COL, DURATION_COL, EVENT_COL]].copy()
    out["cif_up_30d"] = np.clip(cif_up[-1, :], 0.0, 1.0)
    out["cif_down_30d"] = np.clip(cif_down[-1, :], 0.0, 1.0)
    out["p_no_hit_30d"] = np.clip(1.0 - out["cif_up_30d"] - out["cif_down_30d"], 0.0, 1.0)

    return out, times, cif_up, cif_down


def _plot_cif(times: np.ndarray, cif_up: np.ndarray, cif_down: np.ndarray, outdir: Path) -> Path:
    mean_up = cif_up.mean(axis=1)
    mean_down = cif_down.mean(axis=1)
    mean_no_hit = np.clip(1 - (mean_up + mean_down), 0, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, mean_up, label="CIF up", color="#2A9D8F", linewidth=2)
    ax.plot(times, mean_down, label="CIF down", color="#E76F51", linewidth=2)
    ax.plot(times, mean_no_hit, label="No-hit", color="#264653", linewidth=2)
    ax.set_title("Mean CIF curves on test")
    ax.set_xlabel("Days from t0")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    path = outdir / "cif_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple cause-specific Cox baseline")
    p.add_argument("--data-path", type=Path, default=Path("data/gold/survival_dataset.parquet"))
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--valid-ratio", type=float, default=0.15)
    p.add_argument("--train-end", type=str, default=None)
    p.add_argument("--valid-end", type=str, default=None)
    p.add_argument("--min-date", type=str, default=None)
    p.add_argument("--max-date", type=str, default=None)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--penalizer", type=float, default=0.01)
    p.add_argument("--ph-alpha", type=float, default=0.05)
    p.add_argument("--scale-method", type=str, default="train_standard", choices=["train_standard", "none"])
    p.add_argument("--winsor-lower-q", type=float, default=0.005)
    p.add_argument("--winsor-upper-q", type=float, default=0.995)
    p.add_argument("--extreme-abs-threshold", type=float, default=1e4)
    p.add_argument("--outdir", type=Path, default=Path("experiments/ph/outputs"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_gold_dataset(args.data_path, min_date=args.min_date, max_date=args.max_date)
    feature_cols = select_technical_feature_columns(df)

    missing = sorted(set(TECHNICAL_FEATURE_COLS) - set(feature_cols))
    if missing:
        print(f"[warn] missing features: {missing}")

    if not feature_cols:
        raise ValueError("No requested technical features found in dataset.")

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
    print(f"[split] train_end={splits.train_end} valid_end={splits.valid_end}")
    print(
        f"[data-quality] before_sanitize inf="
        f"{splits.quality_report['before_sanitize']['train']['inf_count']} "
        f"nan={splits.quality_report['before_sanitize']['train']['nan_count']} "
        f"abs_gt={splits.quality_report['before_sanitize']['train']['abs_gt_threshold_count']}"
    )

    up_event_col, down_event_col = "event_up", "event_down"

    up_model = CoxPHFitter(penalizer=args.penalizer)
    down_model = CoxPHFitter(penalizer=args.penalizer)

    up_model.fit(splits.train[splits.feature_cols + [DURATION_COL, up_event_col]], DURATION_COL, up_event_col)
    down_model.fit(splits.train[splits.feature_cols + [DURATION_COL, down_event_col]], DURATION_COL, down_event_col)

    preds: dict[str, dict[str, pd.DataFrame]] = {"up": {}, "down": {}}
    metrics: dict[str, dict[str, dict[str, float]]] = {"up": {}, "down": {}}

    for split_name, split_df in [("train", splits.train), ("valid", splits.valid), ("test", splits.test)]:
        up_pred = _predict_cause(up_model, split_df, splits.feature_cols, up_event_col, "up", args.horizon)
        down_pred = _predict_cause(down_model, split_df, splits.feature_cols, down_event_col, "down", args.horizon)

        preds["up"][split_name] = up_pred
        preds["down"][split_name] = down_pred

        metrics["up"][split_name] = _metrics_for_split(up_pred, up_event_col, "risk_up", f"p_up_by_{args.horizon}d")
        metrics["down"][split_name] = _metrics_for_split(
            down_pred,
            down_event_col,
            "risk_down",
            f"p_down_by_{args.horizon}d",
        )

    up_ph = proportional_hazard_test(
        up_model,
        splits.train[splits.feature_cols + [DURATION_COL, up_event_col]],
        time_transform="rank",
    ).summary.reset_index().rename(columns={"index": "feature", "p": "p_value"})
    down_ph = proportional_hazard_test(
        down_model,
        splits.train[splits.feature_cols + [DURATION_COL, down_event_col]],
        time_transform="rank",
    ).summary.reset_index().rename(columns={"index": "feature", "p": "p_value"})

    up_ph["minus_log10_p"] = -np.log10(np.clip(up_ph["p_value"].values, 1e-300, 1.0))
    down_ph["minus_log10_p"] = -np.log10(np.clip(down_ph["p_value"].values, 1e-300, 1.0))

    up_ph.to_csv(args.outdir / "ph_test_up.csv", index=False)
    down_ph.to_csv(args.outdir / "ph_test_down.csv", index=False)

    _plot_ph_violations(up_ph, "up", args.outdir, args.ph_alpha)
    _plot_ph_violations(down_ph, "down", args.outdir, args.ph_alpha)
    _plot_calibration(preds["up"]["test"], up_event_col, f"p_up_by_{args.horizon}d", "up", args.outdir)
    _plot_calibration(preds["down"]["test"], down_event_col, f"p_down_by_{args.horizon}d", "down", args.outdir)
    _plot_risk_hist(preds["up"]["test"], up_event_col, "risk_up", "up", args.outdir)
    _plot_risk_hist(preds["down"]["test"], down_event_col, "risk_down", "down", args.outdir)
    _plot_hr(up_model, "up", args.outdir)
    _plot_hr(down_model, "down", args.outdir)

    cif_30d_df, times, cif_up, cif_down = _compute_cif_curves(
        up_model,
        down_model,
        splits.test,
        splits.feature_cols,
        args.horizon,
    )
    _plot_cif(times, cif_up, cif_down, args.outdir)

    test_join = preds["up"]["test"].merge(
        preds["down"]["test"][["ticker", DATE_COL, "risk_down", f"p_down_by_{args.horizon}d"]],
        on=["ticker", DATE_COL],
        how="inner",
    )
    test_join = test_join.merge(
        cif_30d_df[["ticker", DATE_COL, "cif_up_30d", "cif_down_30d", "p_no_hit_30d"]],
        on=["ticker", DATE_COL],
        how="left",
    )

    hit_mask = test_join[EVENT_COL].isin([1, 2]).values
    dir_acc = float(
        np.mean(
            (
                (test_join.loc[hit_mask, "cif_up_30d"].values >= test_join.loc[hit_mask, "cif_down_30d"].values)
                == (test_join.loc[hit_mask, EVENT_COL].values == 1)
            )
        )
    ) if np.any(hit_mask) else float("nan")

    summary = {
        "config": {
            "data_path": str(args.data_path),
            "horizon": args.horizon,
            "penalizer": args.penalizer,
            "scale_method": args.scale_method,
            "ph_alpha": args.ph_alpha,
            "winsor_lower_q": args.winsor_lower_q,
            "winsor_upper_q": args.winsor_upper_q,
            "extreme_abs_threshold": args.extreme_abs_threshold,
            "train_end": str(splits.train_end),
            "valid_end": str(splits.valid_end),
            "features_used": splits.feature_cols,
        },
        "metrics": metrics,
        "data_quality": splits.quality_report,
        "ph_violations": {
            "up": int((up_ph["p_value"] < args.ph_alpha).sum()),
            "down": int((down_ph["p_value"] < args.ph_alpha).sum()),
        },
        "overall": {
            "test_direction_accuracy_on_hit": dir_acc,
            "mean_test_cif_up_30d": float(test_join["cif_up_30d"].mean()),
            "mean_test_cif_down_30d": float(test_join["cif_down_30d"].mean()),
            "mean_test_p_no_hit_30d": float(test_join["p_no_hit_30d"].mean()),
        },
        "note": "Cause-specific hazards are converted to CIF via numerical integration for 30-day probabilities.",
    }

    (args.outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["up"]).T.to_csv(args.outdir / "metrics_up.csv", index=True)
    pd.DataFrame(metrics["down"]).T.to_csv(args.outdir / "metrics_down.csv", index=True)

    test_join.to_parquet(args.outdir / "test_predictions.parquet", index=False)
    cif_30d_df.to_parquet(args.outdir / "cif_30d_test.parquet", index=False)

    print(f"[done] outdir={args.outdir}")
    print(f"[done] PH violations: up={(up_ph['p_value'] < args.ph_alpha).sum()} down={(down_ph['p_value'] < args.ph_alpha).sum()}")
    print(f"[done] test c-index up={metrics['up']['test']['c_index']:.4f} down={metrics['down']['test']['c_index']:.4f}")


if __name__ == "__main__":
    main()
