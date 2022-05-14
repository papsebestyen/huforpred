from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
import logging
import huspacy

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y-%m-%d %H:%M:%S")

if "hu_core_news_lg" not in huspacy.util.get_installed_models():
    huspacy.download()


nlp = huspacy.load()

df_messages = pd.read_parquet(Path("output/processed_msg.parquet"))

spacy_pipeline = nlp.pipe(
    df_messages["text"],
    disable=[
        "tok2vec",
        "senter",
        "tagger",
        "morphologizer",
        "lemmatizer",
        "parser",
    ],
    n_process=40,
    batch_size=2_000,
)

logging.warning("Preprocess texts wtih Spacy")
docs = [doc for doc in tqdm(spacy_pipeline, total=df_messages["text"].shape[0])]

logging.warning("Get tokens")
tokens = [
    [tok.lower_ for tok in doc if not tok.is_punct and not tok.is_space]
    for doc in tqdm(docs)
]
logging.warning("Get entities")
entities = [[(ent.text, ent.label_) for ent in doc.ents] for doc in tqdm(docs)]

Path("output/tokens.pickle").write_bytes(pickle.dumps(tokens))
Path("output/entities.pickle").write_bytes(pickle.dumps(entities))
logging.warning("Preprocessing is done")
