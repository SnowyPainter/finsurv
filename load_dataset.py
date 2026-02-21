from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path("data/gold/survival_dataset.parquet")
DATE_COL = "Date"
DURATION_COL = "duration"
EVENT_COL = "event"

NON_FEATURE_COLS = {
    "ticker",
    DATE_COL,
    DURATION_COL,
    EVENT_COL,
}

TECHNICAL_FEATURE_COLS = [
    "logret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "vol_60",
    "dist_sma_20",
    "dist_sma_100",
    "volm_z_20",
    "volm_z_60",
    "hl_range",
    "oc_return",
    "atrp_14",
    "rsi_14",
    "bb_width_20",
    "bb_pos_20",
    "macd_hist",
    "obv_ret_5",
    "mom_10_20",
]


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    quality_report: dict[str, dict[str, float | int | str]]


def _apply_train_standard_scaler(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = train[feature_cols].mean()
    stds = train[feature_cols].std(ddof=0)
    # Keep constant columns as zero-centered only.
    stds = stds.replace(0, np.nan)

    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    train.loc[:, feature_cols] = (train[feature_cols] - means) / stds
    valid.loc[:, feature_cols] = (valid[feature_cols] - means) / stds
    test.loc[:, feature_cols] = (test[feature_cols] - means) / stds

    for frame in (train, valid, test):
        frame.loc[:, feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return train, valid, test


def _split_quality_stats(
    frame: pd.DataFrame,
    feature_cols: list[str],
    extreme_abs_threshold: float,
) -> dict[str, float | int]:
    if not feature_cols:
        return {
            "rows": int(len(frame)),
            "nan_count": 0,
            "inf_count": 0,
            "abs_gt_threshold_count": 0,
            "max_abs": 0.0,
        }

    arr = frame[feature_cols].to_numpy(dtype=float, copy=True)
    finite_mask = np.isfinite(arr)
    abs_arr = np.abs(np.where(finite_mask, arr, np.nan))
    finite_abs = abs_arr[np.isfinite(abs_arr)]
    max_abs = float(finite_abs.max()) if finite_abs.size > 0 else float("nan")

    return {
        "rows": int(len(frame)),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "abs_gt_threshold_count": int(np.nansum(abs_arr > extreme_abs_threshold)),
        "max_abs": max_abs,
    }


def _quality_report(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    extreme_abs_threshold: float,
    phase: str,
) -> dict[str, dict[str, float | int | str]]:
    return {
        "phase": {"name": phase},
        "train": _split_quality_stats(train, feature_cols, extreme_abs_threshold),
        "valid": _split_quality_stats(valid, feature_cols, extreme_abs_threshold),
        "test": _split_quality_stats(test, feature_cols, extreme_abs_threshold),
    }


def _sanitize_inf(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> None:
    for frame in (train, valid, test):
        frame.loc[:, feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)


def _apply_train_winsor_clip(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    lower_q: float,
    upper_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lower = train[feature_cols].quantile(lower_q)
    upper = train[feature_cols].quantile(upper_q)

    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    train.loc[:, feature_cols] = train[feature_cols].clip(lower=lower, upper=upper, axis=1)
    valid.loc[:, feature_cols] = valid[feature_cols].clip(lower=lower, upper=upper, axis=1)
    test.loc[:, feature_cols] = test[feature_cols].clip(lower=lower, upper=upper, axis=1)
    return train, valid, test


def _to_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts


def load_gold_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True)

    min_ts = _to_timestamp(min_date)
    max_ts = _to_timestamp(max_date)

    if min_ts is not None:
        df = df.loc[df[DATE_COL] >= min_ts]
    if max_ts is not None:
        df = df.loc[df[DATE_COL] <= max_ts]

    return df.sort_values([DATE_COL, "ticker"]).reset_index(drop=True)


def infer_feature_columns(
    df: pd.DataFrame,
    extra_exclude: Iterable[str] | None = None,
) -> list[str]:
    exclude = set(NON_FEATURE_COLS)
    if extra_exclude is not None:
        exclude.update(extra_exclude)

    feature_cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def select_technical_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in TECHNICAL_FEATURE_COLS
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def _split_dates_by_ratio(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = np.array(sorted(df[DATE_COL].unique()))
    n_dates = len(unique_dates)
    if n_dates < 3:
        raise ValueError("Need at least 3 unique dates for train/valid/test split.")

    train_idx = max(1, min(int(n_dates * train_ratio), n_dates - 2))
    valid_idx = max(train_idx + 1, min(int(n_dates * (train_ratio + valid_ratio)), n_dates - 1))

    train_end = pd.Timestamp(unique_dates[train_idx - 1])
    valid_end = pd.Timestamp(unique_dates[valid_idx - 1])
    return train_end, valid_end


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    train_end: str | None = None,
    valid_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be in (0, 1)")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1")

    train_end_ts = _to_timestamp(train_end)
    valid_end_ts = _to_timestamp(valid_end)

    if train_end_ts is None or valid_end_ts is None:
        inferred_train_end, inferred_valid_end = _split_dates_by_ratio(df, train_ratio, valid_ratio)
        train_end_ts = train_end_ts or inferred_train_end
        valid_end_ts = valid_end_ts or inferred_valid_end

    if train_end_ts >= valid_end_ts:
        raise ValueError("train_end must be earlier than valid_end")

    train = df.loc[df[DATE_COL] <= train_end_ts].copy()
    valid = df.loc[(df[DATE_COL] > train_end_ts) & (df[DATE_COL] <= valid_end_ts)].copy()
    test = df.loc[df[DATE_COL] > valid_end_ts].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError("One of train/valid/test split is empty. Adjust split ratios or dates.")

    return train, valid, test, train_end_ts, valid_end_ts


def add_cause_specific_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_up"] = (out[EVENT_COL] == 1).astype(int)
    out["event_down"] = (out[EVENT_COL] == 2).astype(int)
    return out


def clean_for_model(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cols = feature_cols + [DATE_COL, "ticker", DURATION_COL, EVENT_COL, "event_up", "event_down"]
    existing = [c for c in cols if c in df.columns]
    out = df[existing].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def prepare_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    train_end: str | None = None,
    valid_end: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    feature_cols: list[str] | None = None,
    extra_exclude: Iterable[str] | None = None,
    scale_method: str = "train_standard",
    winsor_lower_q: float = 0.005,
    winsor_upper_q: float = 0.995,
    extreme_abs_threshold: float = 1e4,
) -> DatasetSplits:
    df = load_gold_dataset(data_path, min_date=min_date, max_date=max_date)

    selected_features = feature_cols or infer_feature_columns(df, extra_exclude=extra_exclude)
    selected_features = [c for c in selected_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not selected_features:
        raise ValueError("No usable numeric feature columns found.")

    train, valid, test, train_end_ts, valid_end_ts = time_split(
        df,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        train_end=train_end,
        valid_end=valid_end,
    )

    train = add_cause_specific_targets(train)
    valid = add_cause_specific_targets(valid)
    test = add_cause_specific_targets(test)

    before_report = _quality_report(
        train,
        valid,
        test,
        selected_features,
        extreme_abs_threshold=extreme_abs_threshold,
        phase="before_sanitize",
    )

    _sanitize_inf(train, valid, test, selected_features)

    if not (0.0 <= winsor_lower_q < winsor_upper_q <= 1.0):
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")

    if winsor_lower_q > 0.0 or winsor_upper_q < 1.0:
        train, valid, test = _apply_train_winsor_clip(
            train,
            valid,
            test,
            selected_features,
            lower_q=winsor_lower_q,
            upper_q=winsor_upper_q,
        )

    after_winsor_report = _quality_report(
        train,
        valid,
        test,
        selected_features,
        extreme_abs_threshold=extreme_abs_threshold,
        phase="after_winsor",
    )

    rows_before_clean = {"train": len(train), "valid": len(valid), "test": len(test)}

    train = clean_for_model(train, selected_features)
    valid = clean_for_model(valid, selected_features)
    test = clean_for_model(test, selected_features)

    if scale_method == "train_standard":
        train, valid, test = _apply_train_standard_scaler(train, valid, test, selected_features)
    elif scale_method != "none":
        raise ValueError("scale_method must be one of: train_standard, none")

    # Re-check empties after NA/inf cleanup.
    if train.empty or valid.empty or test.empty:
        raise ValueError("One split became empty after cleaning. Reduce feature set or date filters.")

    after_scale_report = _quality_report(
        train,
        valid,
        test,
        selected_features,
        extreme_abs_threshold=extreme_abs_threshold,
        phase="after_scale",
    )

    rows_after_clean = {"train": len(train), "valid": len(valid), "test": len(test)}

    quality_report: dict[str, dict[str, float | int | str]] = {
        "before_sanitize": before_report,
        "after_winsor": after_winsor_report,
        "after_scale": after_scale_report,
        "rows": {
            "train_before_clean": rows_before_clean["train"],
            "valid_before_clean": rows_before_clean["valid"],
            "test_before_clean": rows_before_clean["test"],
            "train_after_clean": rows_after_clean["train"],
            "valid_after_clean": rows_after_clean["valid"],
            "test_after_clean": rows_after_clean["test"],
        },
        "winsor_config": {
            "winsor_lower_q": winsor_lower_q,
            "winsor_upper_q": winsor_upper_q,
            "extreme_abs_threshold": extreme_abs_threshold,
        },
    }

    return DatasetSplits(
        train=train,
        valid=valid,
        test=test,
        feature_cols=selected_features,
        train_end=train_end_ts,
        valid_end=valid_end_ts,
        quality_report=quality_report,
    )


def _event_table(df: pd.DataFrame) -> dict[str, int]:
    s = df[EVENT_COL].value_counts().to_dict()
    return {
        "censor(0)": int(s.get(0, 0)),
        "up(1)": int(s.get(1, 0)),
        "down(2)": int(s.get(2, 0)),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load and split survival dataset for experiments.")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--valid-ratio", type=float, default=0.15)
    p.add_argument("--train-end", type=str, default=None)
    p.add_argument("--valid-end", type=str, default=None)
    p.add_argument("--min-date", type=str, default=None)
    p.add_argument("--max-date", type=str, default=None)
    p.add_argument("--max-features", type=int, default=None, help="Use first N inferred features for quick tests")
    p.add_argument("--scale-method", type=str, default="train_standard", choices=["train_standard", "none"])
    p.add_argument("--winsor-lower-q", type=float, default=0.005)
    p.add_argument("--winsor-upper-q", type=float, default=0.995)
    p.add_argument("--extreme-abs-threshold", type=float, default=1e4)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    df = load_gold_dataset(args.data_path, min_date=args.min_date, max_date=args.max_date)
    features = infer_feature_columns(df)
    if args.max_features is not None:
        features = features[: args.max_features]

    splits = prepare_dataset(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        train_end=args.train_end,
        valid_end=args.valid_end,
        min_date=args.min_date,
        max_date=args.max_date,
        feature_cols=features,
        scale_method=args.scale_method,
        winsor_lower_q=args.winsor_lower_q,
        winsor_upper_q=args.winsor_upper_q,
        extreme_abs_threshold=args.extreme_abs_threshold,
    )

    for name, frame in [
        ("train", splits.train),
        ("valid", splits.valid),
        ("test", splits.test),
    ]:
        print(
            f"[{name}] rows={len(frame):,} "
            f"date={frame[DATE_COL].min().date()}..{frame[DATE_COL].max().date()} "
            f"events={_event_table(frame)}"
        )

    print(f"features={len(splits.feature_cols)}")
    print(f"train_end={splits.train_end}")
    print(f"valid_end={splits.valid_end}")
    print("[quality_report]")
    print(json.dumps(splits.quality_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
