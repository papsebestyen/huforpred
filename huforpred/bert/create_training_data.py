from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

processed_data_path = Path("../../data/processed")
training_data_path = Path("../../data/training")

df_messages = pd.read_parquet(processed_data_path / "processed_msg.parquet")
stock_data = pd.read_parquet(processed_data_path / "stock_data.parquet")
movement = (stock_data["Close"].diff(-1) * -1 > 0).map({True: 1, False: 0})
movement.index = pd.to_datetime(movement.index)

df_classification = (
    df_messages.assign(label=lambda _df: _df["date"].dt.date.map(movement))
    .dropna(subset=["label"])
    .sample(2_000, random_state=42)[["label", "text"]]
    .reset_index(drop=True)
    .astype({"label": int})
)

df_train, df_test = train_test_split(
    df_classification,
    train_size=0.8,
    stratify=df_classification["label"],
    random_state=42,
)

df_train.to_parquet(training_data_path / "labelled_train.parquet", index=False)
df_test.to_parquet(training_data_path / "labelled_test.parquet", index=False)
