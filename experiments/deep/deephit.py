from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_ipcw
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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


class DeepHitMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, horizon: int, causes: int = 2) -> None:
        super().__init__()
        self.horizon = horizon
        self.causes = causes
        out_dim = 1 + causes * horizon  # class 0: no-hit by horizon, then (cause,time) bins
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_targets(duration: np.ndarray, event: np.ndarray, horizon: int) -> np.ndarray:
    d = np.clip(duration.astype(int), 1, horizon)
    e = event.astype(int)

    y = np.zeros(len(d), dtype=np.int64)  # 0 = no-hit by horizon
    hit = np.isin(e, [1, 2])
    y[hit] = 1 + (e[hit] - 1) * horizon + (d[hit] - 1)
    return y


def extract_probs(prob: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # prob shape: [n, 1 + 2*horizon]
    cause_time = prob[:, 1:].reshape(prob.shape[0], 2, horizon)
    pmf_up = cause_time[:, 0, :]
    pmf_down = cause_time[:, 1, :]
    cif_up = torch.cumsum(pmf_up, dim=1)
    cif_down = torch.cumsum(pmf_down, dim=1)
    p_no_hit = prob[:, 0]
    return pmf_up, pmf_down, cif_up, cif_down, p_no_hit


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
        l, r = edges[i], edges[i + 1]
        m = (p >= l) & (p < r if i < bins - 1 else p <= r)
        if not np.any(m):
            continue
        ece += abs(float(np.mean(y[m])) - float(np.mean(p[m]))) * (np.sum(m) / len(y))
    return float(ece)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) <= 1:
        return float("nan")
    return float(roc_auc_score(y, p))


def _to_surv_struct(duration: np.ndarray, event_binary: np.ndarray) -> np.ndarray:
    return np.array(
        list(zip(event_binary.astype(bool), duration.astype(float), strict=False)),
        dtype=[("event", "?"), ("time", "<f8")],
    )


@dataclass
class IPCWCache:
    up_train_surv: np.ndarray
    down_train_surv: np.ndarray


def build_ipcw_cache(train_duration: np.ndarray, train_event: np.ndarray) -> IPCWCache:
    return IPCWCache(
        up_train_surv=_to_surv_struct(train_duration, (train_event == 1).astype(int)),
        down_train_surv=_to_surv_struct(train_duration, (train_event == 2).astype(int)),
    )


def uno_c_index(
    ipcw_cache: IPCWCache,
    eval_duration: np.ndarray,
    eval_event: np.ndarray,
    risk_score: np.ndarray,
    cause: int,
    horizon: int,
) -> float:
    y_train = ipcw_cache.up_train_surv if cause == 1 else ipcw_cache.down_train_surv
    y_eval = _to_surv_struct(eval_duration.astype(float), (eval_event.astype(int) == cause).astype(int))

    tau = float(horizon)
    cidx, *_ = concordance_index_ipcw(y_train, y_eval, risk_score.astype(float), tau=tau)
    return float(cidx)


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


def split_metrics(
    split: dict[str, Any],
    prob: torch.Tensor,
    horizon: int,
    ipcw_cache: IPCWCache,
    compute_uno: bool,
    use_tqdm: bool,
    split_name: str = "",
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    _, _, cif_up_t, cif_down_t, p_no_hit_t = extract_probs(prob, horizon)

    # Move CIF tensors to CPU once to avoid repeated GPU->CPU sync in loops.
    cif_up = cif_up_t.cpu().numpy()
    cif_down = cif_down_t.cpu().numpy()
    p_no_h = np.clip(p_no_hit_t.cpu().numpy(), 0.0, 1.0)
    p_up_h = cif_up[:, horizon - 1]
    p_down_h = cif_down[:, horizon - 1]

    y_3 = split["event"]
    duration = split["duration"]
    y_up = (y_3 == 1).astype(int)
    y_down = (y_3 == 2).astype(int)

    pred_3 = np.argmax(np.column_stack([p_no_h, p_up_h, p_down_h]), axis=1)

    up_top, up_lift = _top_decile(p_up_h, y_up)
    down_top, down_lift = _top_decile(p_down_h, y_down)

    if compute_uno:
        uno_iter = tqdm(
            total=2,
            desc=f"Uno {split_name}",
            dynamic_ncols=True,
            leave=False,
            disable=not use_tqdm,
        )
        uno_up = uno_c_index(ipcw_cache, duration, y_3, p_up_h, cause=1, horizon=horizon)
        if use_tqdm:
            uno_iter.update(1)
            uno_iter.set_postfix(uno_up=f"{uno_up:.3f}")

        uno_down = uno_c_index(ipcw_cache, duration, y_3, p_down_h, cause=2, horizon=horizon)
        if use_tqdm:
            uno_iter.update(1)
            uno_iter.set_postfix(uno_down=f"{uno_down:.3f}")
            uno_iter.close()
    else:
        uno_up = float("nan")
        uno_down = float("nan")

    metrics = {
        "n": int(len(y_3)),
        "event_rate_up": float(np.mean(y_up)),
        "event_rate_down": float(np.mean(y_down)),
        "mean_p_up_30d": float(np.mean(p_up_h)),
        "mean_p_down_30d": float(np.mean(p_down_h)),
        "mean_p_no_hit_30d": float(np.mean(p_no_h)),
        "sum_prob_le_1_rate": float(np.mean((p_up_h + p_down_h) <= 1.000001)),
        "brier_up": float(np.mean((p_up_h - y_up) ** 2)),
        "brier_down": float(np.mean((p_down_h - y_down) ** 2)),
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

    pred_df = pd.DataFrame(
        {
            "ticker": split["ticker"],
            DATE_COL: split["date"],
            DURATION_COL: duration,
            EVENT_COL: y_3,
        }
    )
    pred_df["p_up_30d"] = p_up_h
    pred_df["p_down_30d"] = p_down_h
    pred_df["p_no_hit_30d"] = np.clip(1 - p_up_h - p_down_h, 0.0, 1.0)
    pred_df["pred_3class"] = pred_3

    return metrics, pred_df, _confusion_3class(y_3, pred_3)


def time_dependent_metrics(
    split: dict[str, Any],
    prob: torch.Tensor,
    horizon: int,
    td_times: list[int],
    split_name: str,
    use_tqdm: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    _, _, cif_up_t, cif_down_t, _ = extract_probs(prob, horizon)
    # Move CIF tensors to CPU once to avoid repeated GPU->CPU sync in loops.
    cif_up = cif_up_t.cpu().numpy()
    cif_down = cif_down_t.cpu().numpy()
    duration = split["duration"]
    event = split["event"]

    rows: list[dict[str, float | int]] = []
    time_iter = tqdm(
        td_times,
        desc=f"TD {split_name}",
        dynamic_ncols=True,
        leave=False,
        disable=not use_tqdm,
    )
    for t in time_iter:
        y_up_t = ((event == 1) & (duration <= t)).astype(int)
        y_down_t = ((event == 2) & (duration <= t)).astype(int)

        p_up_t = cif_up[:, t - 1]
        p_down_t = cif_down[:, t - 1]
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
    last_t = int(max(td_times))
    summary = {
        "ibs_up": float(td_df["brier_up"].mean()),
        "ibs_down": float(td_df["brier_down"].mean()),
        "ibs_overall": float(td_df["brier_overall"].mean()),
        "brier_up_30d": float(td_df.loc[td_df["t"] == last_t, "brier_up"].iloc[0]),
        "brier_down_30d": float(td_df.loc[td_df["t"] == last_t, "brier_down"].iloc[0]),
        "brier_overall_30d": float(td_df.loc[td_df["t"] == last_t, "brier_overall"].iloc[0]),
        "td_auc_up_mean": float(td_df["auc_up"].dropna().mean()) if td_df["auc_up"].notna().any() else float("nan"),
        "td_auc_down_mean": float(td_df["auc_down"].dropna().mean()) if td_df["auc_down"].notna().any() else float("nan"),
    }
    return summary, td_df


def plot_calibration(pred_df: pd.DataFrame, cause: str, outdir: Path) -> Path:
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


def plot_cif_curves(cif_up: np.ndarray, cif_down: np.ndarray, outdir: Path) -> Path:
    t = np.arange(1, cif_up.shape[1] + 1)
    mean_up = cif_up.mean(axis=0)
    mean_down = cif_down.mean(axis=0)
    no_hit = np.clip(1 - mean_up - mean_down, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean_up, label="CIF up")
    ax.plot(t, mean_down, label="CIF down")
    ax.plot(t, no_hit, label="No-hit")
    ax.set_xlabel("Day")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Mean CIF curves")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    path = outdir / "cif_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prob_dist(pred_df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    ax.hist(pred_df["p_up_30d"], bins=bins, alpha=0.6, label="p_up_30d")
    ax.hist(pred_df["p_down_30d"], bins=bins, alpha=0.6, label="p_down_30d")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted CIF distribution @30d")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = outdir / "cif_30d_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(cm: np.ndarray, outdir: Path) -> Path:
    labels = ["no_hit", "up", "down"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("3-class confusion (test)")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    path = outdir / "confusion_3class_test.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_time_dependent_curves(td_df: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(td_df["t"], td_df["auc_up"], label="AUC up")
    axes[0].plot(td_df["t"], td_df["auc_down"], label="AUC down")
    axes[0].set_title("Time-dependent AUC")
    axes[0].set_xlabel("t (day)")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].plot(td_df["t"], td_df["brier_up"], label="Brier up")
    axes[1].plot(td_df["t"], td_df["brier_down"], label="Brier down")
    axes[1].plot(td_df["t"], td_df["brier_overall"], label="Brier overall")
    axes[1].set_title("Brier by time (IBS source)")
    axes[1].set_xlabel("t (day)")
    axes[1].set_ylabel("Brier score")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")
    fig.tight_layout()

    path = outdir / "time_dependent_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@dataclass
class TrainOutput:
    model: DeepHitMLP
    best_valid_loss: float


def fit_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    seed: int,
    use_tqdm: bool,
) -> TrainOutput:
    set_seed(seed)

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    x_valid = valid_df[feature_cols].to_numpy(dtype=np.float32)

    y_train = build_targets(train_df[DURATION_COL].values, train_df[EVENT_COL].values, horizon)
    y_valid = build_targets(valid_df[DURATION_COL].values, valid_df[EVENT_COL].values, horizon)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    valid_ds = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepHitMLP(
        in_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        dropout=dropout,
        horizon=horizon,
        causes=2,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=1 + 2 * horizon)
    class_weights = np.zeros_like(class_counts, dtype=np.float32)
    nonzero = class_counts > 0
    class_weights[nonzero] = len(y_train) / (len(class_counts) * class_counts[nonzero])
    class_weights = np.clip(class_weights, 0.2, 10.0)

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    epoch_iter = tqdm(
        range(1, max_epochs + 1),
        desc="Epoch",
        dynamic_ncols=True,
        leave=True,
        disable=not use_tqdm,
    )
    for epoch in epoch_iter:
        model.train()
        tr_losses = []
        train_iter = tqdm(
            train_loader,
            desc=f"Train {epoch}",
            dynamic_ncols=True,
            leave=False,
            disable=not use_tqdm,
        )
        for xb, yb in train_iter:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))
            if use_tqdm:
                train_iter.set_postfix(loss=f"{tr_losses[-1]:.4f}")

        model.eval()
        va_losses = []
        with torch.no_grad():
            valid_iter = tqdm(
                valid_loader,
                desc=f"Valid {epoch}",
                dynamic_ncols=True,
                leave=False,
                disable=not use_tqdm,
            )
            for xb, yb in valid_iter:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                vloss = float(criterion(logits, yb).item())
                va_losses.append(vloss)
                if use_tqdm:
                    valid_iter.set_postfix(loss=f"{vloss:.4f}")

        valid_loss = float(np.mean(va_losses))
        train_loss = float(np.mean(tr_losses))
        if use_tqdm:
            epoch_iter.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}", best=f"{best_loss:.4f}")
        if epoch == 1 or epoch % 10 == 0:
            print(f"[train] epoch={epoch} train_loss={np.mean(tr_losses):.4f} valid_loss={valid_loss:.4f}")

        if valid_loss + 1e-6 < best_loss:
            best_loss = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[train] early stop at epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce any checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    return TrainOutput(model=model, best_valid_loss=best_loss)


def predict_prob(
    model: DeepHitMLP,
    frame: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int,
    use_tqdm: bool,
) -> torch.Tensor:
    x = frame[feature_cols].to_numpy(dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = next(model.parameters()).device
    outs: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        pred_iter = tqdm(loader, desc="Predict", dynamic_ncols=True, leave=False, disable=not use_tqdm)
        for (xb,) in pred_iter:
            xb = xb.to(device)
            prob = torch.softmax(model(xb), dim=1).cpu()
            outs.append(prob)

    return torch.cat(outs, dim=0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepHit-style competing-risks model (event 1&2 jointly)")
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--td-times", type=str, default="10,30")
    p.add_argument(
        "--postprocess-splits",
        type=str,
        default="test",
        help="Comma-separated splits to compute postprocess metrics for (subset of train,valid,test).",
    )
    p.add_argument(
        "--uno-splits",
        type=str,
        default="test",
        help="Comma-separated splits to run Uno C-index on (subset of postprocess splits).",
    )
    p.add_argument("--force-retrain", action="store_true", help="Ignore existing model.pt and retrain")
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--outdir", type=Path, default=Path("experiments/deep/outputs/deephit"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    use_tqdm = not args.no_tqdm
    td_times = sorted(
        {
            t
            for t in (int(x.strip()) for x in args.td_times.split(",") if x.strip())
            if 1 <= t <= args.horizon
        }
    )
    if not td_times:
        raise ValueError("--td-times must include at least one integer within [1, horizon].")

    valid_split_names = {"train", "valid", "test"}
    postprocess_splits = [s.strip().lower() for s in args.postprocess_splits.split(",") if s.strip()]
    if not postprocess_splits:
        raise ValueError("--postprocess-splits must include at least one split name.")
    if any(s not in valid_split_names for s in postprocess_splits):
        raise ValueError("--postprocess-splits must be a subset of: train,valid,test")
    if "test" not in postprocess_splits:
        raise ValueError("--postprocess-splits must include test (required for plots and outputs).")
    postprocess_splits = list(dict.fromkeys(postprocess_splits))

    uno_splits = [s.strip().lower() for s in args.uno_splits.split(",") if s.strip()]
    if any(s not in valid_split_names for s in uno_splits):
        raise ValueError("--uno-splits must be a subset of: train,valid,test")
    if any(s not in postprocess_splits for s in uno_splits):
        raise ValueError("--uno-splits must be a subset of --postprocess-splits")
    uno_splits = set(uno_splits)

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

    model_path = args.outdir / "model.pt"
    if model_path.exists() and not args.force_retrain:
        ckpt = torch.load(model_path, map_location="cpu")
        ckpt_features = ckpt.get("feature_cols")
        if ckpt_features is not None and list(ckpt_features) != list(splits.feature_cols):
            raise ValueError("Existing model feature columns do not match current dataset features. Use --force-retrain.")
        if int(ckpt.get("horizon", args.horizon)) != args.horizon:
            raise ValueError("Existing model horizon does not match current --horizon. Use --force-retrain.")
        if int(ckpt.get("hidden_dim", args.hidden_dim)) != args.hidden_dim:
            raise ValueError("Existing model hidden_dim does not match current --hidden-dim. Use --force-retrain.")
        if float(ckpt.get("dropout", args.dropout)) != float(args.dropout):
            raise ValueError("Existing model dropout does not match current --dropout. Use --force-retrain.")

        model = DeepHitMLP(
            in_dim=len(splits.feature_cols),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            horizon=args.horizon,
            causes=2,
        )
        model.load_state_dict(ckpt["state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        out = TrainOutput(model=model, best_valid_loss=float(ckpt.get("best_valid_loss", float("nan"))))
        print(f"[fit] loaded existing model: {model_path}")
    else:
        out = fit_model(
            train_df=splits.train,
            valid_df=splits.valid,
            feature_cols=splits.feature_cols,
            horizon=args.horizon,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            weight_decay=args.weight_decay,
            seed=args.seed,
            use_tqdm=use_tqdm,
        )
        torch.save(
            {
                "state_dict": out.model.state_dict(),
                "feature_cols": splits.feature_cols,
                "horizon": args.horizon,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_valid_loss": out.best_valid_loss,
            },
            model_path,
        )
        print(f"[fit] saved model: {model_path}")

    frames_by_split = {
        "train": splits.train,
        "valid": splits.valid,
        "test": splits.test,
    }

    t0 = time.perf_counter()
    probs = {
        split_name: predict_prob(out.model, frames_by_split[split_name], splits.feature_cols, args.batch_size, use_tqdm=use_tqdm)
        for split_name in postprocess_splits
    }
    print(f"[time] prediction={time.perf_counter() - t0:.1f}s")

    split_arrays: dict[str, dict[str, Any]] = {
        name: {
            "ticker": frame["ticker"].to_numpy(),
            "date": frame[DATE_COL].to_numpy(),
            "duration": frame[DURATION_COL].to_numpy(dtype=int),
            "event": frame[EVENT_COL].to_numpy(dtype=int),
        }
        for name, frame in frames_by_split.items()
    }
    ipcw_cache = build_ipcw_cache(split_arrays["train"]["duration"], split_arrays["train"]["event"])

    metrics: dict[str, dict[str, float]] = {}
    preds: dict[str, pd.DataFrame] = {}
    cms: dict[str, np.ndarray] = {}
    td_summary: dict[str, dict[str, float]] = {}
    td_frames: dict[str, pd.DataFrame] = {}

    split_items = [(name, None) for name in postprocess_splits]
    split_iter = tqdm(
        split_items,
        desc="Postprocess Split",
        dynamic_ncols=True,
        leave=True,
        disable=not use_tqdm,
    )
    t1 = time.perf_counter()
    for split_name, _frame in split_iter:
        if use_tqdm:
            split_iter.set_postfix(step=f"{split_name}:start")
        m, p, cm = split_metrics(
            split_arrays[split_name],
            probs[split_name],
            args.horizon,
            ipcw_cache,
            compute_uno=split_name in uno_splits,
            use_tqdm=use_tqdm,
            split_name=split_name,
        )
        td_m, td_df = time_dependent_metrics(
            split_arrays[split_name],
            probs[split_name],
            args.horizon,
            td_times,
            split_name,
            use_tqdm=use_tqdm,
        )
        if use_tqdm:
            split_iter.set_postfix(step=f"{split_name}:done", uno_up=f"{m['uno_c_up']:.3f}", uno_down=f"{m['uno_c_down']:.3f}")
        m.update(td_m)
        metrics[split_name] = m
        preds[split_name] = p
        cms[split_name] = cm
        td_summary[split_name] = td_m
        td_frames[split_name] = td_df
    print(f"[time] postprocess={time.perf_counter() - t1:.1f}s")

    _, _, cif_up_test_t, cif_down_test_t, _ = extract_probs(probs["test"], args.horizon)
    cif_up_test = cif_up_test_t.cpu().numpy()
    cif_down_test = cif_down_test_t.cpu().numpy()

    t2 = time.perf_counter()
    plot_calibration(preds["test"], "up", args.outdir)
    plot_calibration(preds["test"], "down", args.outdir)
    plot_cif_curves(cif_up_test, cif_down_test, args.outdir)
    plot_prob_dist(preds["test"], args.outdir)
    plot_confusion(cms["test"], args.outdir)
    plot_time_dependent_curves(td_frames["test"], args.outdir)
    print(f"[time] plotting={time.perf_counter() - t2:.1f}s")

    t3 = time.perf_counter()
    preds["test"].to_parquet(args.outdir / "test_predictions.parquet", index=False)
    td_frames["test"].to_csv(args.outdir / "time_dependent_metrics_test.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(args.outdir / "metrics.csv", index=True)
    pd.DataFrame(cms["test"], index=["true_no_hit", "true_up", "true_down"], columns=["pred_no_hit", "pred_up", "pred_down"]).to_csv(
        args.outdir / "confusion_3class_test.csv"
    )

    summary = {
        "config": {
            "data_path": str(args.data_path),
            "horizon": args.horizon,
            "features_used": splits.feature_cols,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "seed": args.seed,
            "td_times": td_times,
            "postprocess_splits": postprocess_splits,
            "uno_splits": sorted(uno_splits),
            "winsor_lower_q": args.winsor_lower_q,
            "winsor_upper_q": args.winsor_upper_q,
            "extreme_abs_threshold": args.extreme_abs_threshold,
            "train_end": str(splits.train_end),
            "valid_end": str(splits.valid_end),
        },
        "data_quality": splits.quality_report,
        "best_valid_loss": out.best_valid_loss,
        "metrics": metrics,
        "notes": [
            "Competing risks are modeled jointly with one softmax over (no-hit, cause,time) bins.",
            "CIF_k(t) is computed by cumulative sum of per-time PMF for cause k.",
        ],
    }
    (args.outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[time] save={time.perf_counter() - t3:.1f}s")

    print(f"[done] outdir={args.outdir}")
    print(
        f"[done] test metrics: acc_3class={metrics['test']['acc_3class']:.4f}, "
        f"macro_f1_3class={metrics['test']['macro_f1_3class']:.4f}, "
        f"lift_up={metrics['test']['top_decile_lift_up']:.3f}, "
        f"lift_down={metrics['test']['top_decile_lift_down']:.3f}, "
        f"ibs_overall={metrics['test']['ibs_overall']:.4f}"
    )


if __name__ == "__main__":
    main()
