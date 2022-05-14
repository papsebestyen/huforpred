from gensim.models.fasttext import load_facebook_model
from gensim.models.callbacks import CallbackAny2Vec
import pickle
from pathlib import Path
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y-%m-%d %H:%M:%S")

logging.warning('Read token file')
stock_toks = pickle.loads(Path('output/tokens.pickle').read_bytes())

class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logging.warning(f"Loss after epoch {self.epoch}: {loss}")
        self.epoch += 1

logging.warning('Load pretrained FastText model')
model = load_facebook_model(Path('../jeszk_moments/hu.szte.w2v.fasttext.bin'))
logging.warning('Update vocabulary')
model.build_vocab(stock_toks, update=True)
model.workers = 40

logging.warning('Train model')
model.train(
    corpus_iterable=stock_toks,
    total_examples=len(stock_toks),
    epochs=model.epochs,
    compute_loss=True,
    callbacks=[callback()],
)

logging.warning('Save model')
model.save('output/language_model/stock_language_model')

logging.warning('FastText training is done')