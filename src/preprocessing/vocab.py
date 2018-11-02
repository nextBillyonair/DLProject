import logging
from glob import glob


SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SOS_INDEX = 0
EOS_INDEX = 1

MODERN_PATH='../../data/modern/'
ORIGINAL_PATH='../../data/original/'
PROCESSED_PATH='../../data/processed/'

class Vocab:
    """This class handles the mapping between the words and their indices."""
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_INDEX: SOS_TOKEN, EOS_INDEX: EOS_TOKEN}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_files(path=PROCESSED_PATH):
    return glob(path+'*')


def read_file(input_file):
    logging.info("Reading lines of %s...", input_file)
    return open(input_file).read().strip().split('\n')


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """
    Creates the vocabs for each of the languages based on the training
    corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = read_file(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info(f'{src_vocab.lang_code} src vocab size: {src_vocab.n_words}')
    logging.info(f'{tgt_vocab.lang_code} tgt vocab size: {tgt_vocab.n_words}')

    return src_vocab, tgt_vocab
