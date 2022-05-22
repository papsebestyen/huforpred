from pathlib import Path
import pandas as pd

url_regex = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"

df_topics = pd.read_parquet(Path('data/raw/topics.parquet'))

bet_threads = df_topics[lambda _df: _df['topic'] == 'bét'].index

df_messages = (
    pd.concat(
        [
            pd.read_parquet(Path(f"data/raw/messages/{thread}.parquet")).assign(
                thread_id=thread
            )
            for thread in bet_threads
        ]
    )
    .assign(text_len=lambda _df: _df["text"].str.len())
    .loc[
        lambda _df: (~_df["text"].isin(["Törölt hozzászólás", "link"]))
        & (_df["text_len"] < 512)
        & (_df["text_len"] > 0)
        & (~_df["text"].str.contains(url_regex)),
        :,
    ]
    .drop(columns=["like", "unlike", "text_len"])
    .assign(
        comment_id=lambda _df: _df["thread_id"].astype(str)
        + "-"
        + _df.index.astype(str)
    )
    .set_index("comment_id", drop=False)
)

df_messages.to_parquet(Path('data/processed/processed_msg.parquet'))