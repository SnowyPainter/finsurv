import kagglehub
import os
import glob
import pandas as pd
from pathlib import Path

path = kagglehub.dataset_download("tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

csv_dir = Path(path)  # kagglehub 다운로드 경로

all_files = glob.glob(str(csv_dir / "*.csv"))

dfs = []

for file in all_files:
    ticker = Path(file).stem  # A.csv -> A
    df = pd.read_csv(file)
    df["ticker"] = ticker
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

full_df.to_parquet(RAW_DIR / "all_stocks.parquet", index=False)

print("Saved RAW parquet.")