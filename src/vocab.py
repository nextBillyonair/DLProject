import logging
from glob import glob

"""File for handling vocabulary building"""

# Tokens
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SOS_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2

# Path to processed aligned data
TRAIN_PATH='../data/split/train.snt.aligned'
DEV_PATH='../data/split/dev.snt.aligned'
TEST_PATH='../data/split/test.snt.aligned'


class Vocab:
    """This class handles the mapping between the words and their indices."""
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {SOS_TOKEN: SOS_INDEX,
                           EOS_TOKEN: EOS_INDEX,
                           UNK_TOKEN: UNK_INDEX}
        self.word2count = {SOS_TOKEN: 0, EOS_TOKEN: 0, UNK_TOKEN: 0}
        self.index2word = {SOS_INDEX: SOS_TOKEN,
                           EOS_INDEX: EOS_TOKEN,
                           UNK_INDEX: UNK_TOKEN}
        self.n_words = 3  # Count SOS, EOS, UNK

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

def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)

    # Read the file and split into lines
    lines = open(input_file).read().strip().split('\n')

    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]

    return pairs


def make_vocabs(src_code='Original', tgt_code='Modern'):
    """
    Creates the vocabs for each of the languages based on the training
    corpus.
    """
    src_vocab = Vocab(src_code)
    tgt_vocab = Vocab(tgt_code)

    # to add all files + split_lines(FILE_PATH)
    train_pairs = split_lines(TRAIN_PATH)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info(f'{src_vocab.lang_code} src vocab size: {src_vocab.n_words}')
    logging.info(f'{tgt_vocab.lang_code} tgt vocab size: {tgt_vocab.n_words}')

    return src_vocab, tgt_vocab


if __name__ == '__main__':
    # USEAGE
    src_vocab, tgt_vocab = make_vocabs('Original', 'Modern')

    print(src_vocab.lang_code, src_vocab.n_words)
    print(tgt_vocab.lang_code, tgt_vocab.n_words)

    # 0 : 21079
    print(src_vocab.word2index[SOS_TOKEN], src_vocab.word2count[SOS_TOKEN])
    # 1 : 21079
    print(src_vocab.word2index[EOS_TOKEN], src_vocab.word2count[EOS_TOKEN])
