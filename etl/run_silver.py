import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/raw/all_stocks.parquet")
df = df.sort_values(["ticker", "Date"])

HORIZON = 30
UPPER = 0.10
LOWER = -0.10

rows = []

grouped = df.groupby("ticker", sort=False)
total_windows = sum(max(len(g) - HORIZON, 0) for _, g in grouped)

with tqdm(total=total_windows, desc="Windows") as window_pbar:
    for ticker, g in tqdm(grouped, total=df["ticker"].nunique(), desc="Tickers"):
        g = g.reset_index(drop=True)
        prices = g["Close"].values

        for i in range(len(g) - HORIZON):
            start_price = prices[i]
            future_prices = prices[i+1:i+HORIZON+1]

            returns = (future_prices / start_price) - 1

            hit_up = np.where(returns >= UPPER)[0]
            hit_down = np.where(returns <= LOWER)[0]

            event = 0
            duration = HORIZON

            if len(hit_up) > 0 and len(hit_down) > 0:
                if hit_up[0] < hit_down[0]:
                    event = 1
                    duration = hit_up[0] + 1
                else:
                    event = 2
                    duration = hit_down[0] + 1

            elif len(hit_up) > 0:
                event = 1
                duration = hit_up[0] + 1

            elif len(hit_down) > 0:
                event = 2
                duration = hit_down[0] + 1

            rows.append({
                "ticker": ticker,
                "Date": g.loc[i, "Date"],
                "duration": duration,
                "event": event,
                "start_price": start_price
            })
            window_pbar.update(1)

silver_df = pd.DataFrame(rows)
silver_df.to_parquet(SILVER_DIR / "survival_windows.parquet", index=False)

print("Saved SILVER survival windows.")
