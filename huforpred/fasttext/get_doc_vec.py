from gensim.models.fasttext import FastText
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y-%m-%d %H:%M:%S")

logging.warning("Get tokens")
docs = pickle.loads(Path("output/tokens.pickle").read_bytes())
logging.warning("Get FastText model")
model = FastText.load("output/language_model/stock_language_model")


def get_doc_vector(doc, model):
    return np.nanmean(
        np.array([model.wv.get_vector(word, norm=True) for word in doc]), axis=0
    )


logging.warning("Get messages")
df_messages = pd.read_parquet(Path("output/processed_msg.parquet"))

logging.warning("Get document_embeddings")
df_embed = pd.DataFrame(
    np.array(
        [get_doc_vector(doc, model) for doc in tqdm([d for d in docs if len(d) != 0])]
    ),
    index=df_messages.index[[len(d) != 0 for d in docs]],
)

logging.warning("Writing pickle")
Path("output/doc_embed.pickle").write_bytes(pickle.dumps(df_embed))

logging.warning("DONE")
