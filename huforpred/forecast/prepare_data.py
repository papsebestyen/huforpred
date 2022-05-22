import pandas as pd
from pathlib import Path
import numpy as np

df = (
    pd.read_parquet(Path("../../data/processed/stock_data.parquet"))
    .reset_index()
    .assign(
        Date=lambda _df: pd.PeriodIndex(
            _df["Date"].astype(str), freq="D"
        ).to_timestamp()
    )
    .set_index("Date")
    .to_period("D")
)

forum_data = (
    pd.read_csv("../../data/processed/thread_data.csv")
    .assign(date=lambda _df: pd.to_datetime(_df["date"]).dt.date)
    .assign(
        Date=lambda _df: pd.PeriodIndex(
            _df["date"].astype(str), freq="D"
        ).to_timestamp(),
        day=lambda _df: _df["day"] / 30 * 2 * np.pi,
        month=lambda _df: _df["month"] / 12 * 2 * np.pi,
        dayofweek=lambda _df: _df["dayofweek"] / 7 * 2 * np.pi,
    )
    .rename(columns={"comment_id": "comment_count"})
    .set_index("Date")
    .to_period("D")
)

df = (
    df.join(forum_data)
    .dropna()
    .reset_index()
    .assign(
        Date=lambda _df: pd.PeriodIndex(
            _df["Date"].astype(str), freq="D"
        ).to_timestamp()
    )
    .set_index("Date")
    .to_period("D")
)

df.reset_index().to_parquet("../../data/processed/final_data.parquet")
