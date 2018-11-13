from itertools import islice
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from time import time

class Timer:
    def __init__(self):
        self._start = time()

    def elapsed(self):
        return time() - self._start

def chunk(iterable, n):
    if n is None:
        yield tuple(iterable)
        return

    it = iter(iterable)

    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return

        yield chunk


def corpus_bleu(references, candidates):
    return nltk_corpus_bleu([[sentence.split()] for sentence in references],
                            [sentence.split() for sentence in candidates])