import logging
from glob import glob

# Tokens
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SOS_INDEX = 0
EOS_INDEX = 1

# Path to processed aligned data
MODERN_PATH='../data/processed/modern.snt.aligned'
ORIGINAL_PATH='../data/processed/original.snt.aligned'


class Vocab:
    """This class handles the mapping between the words and their indices."""
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {SOS_TOKEN: SOS_INDEX, EOS_TOKEN: EOS_INDEX}
        self.word2count = {SOS_TOKEN: 0, EOS_TOKEN: 0}
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


def read_file(input_file):
    logging.info("Reading lines of %s...", input_file)
    return open(input_file).read().strip().split('\n')


def make_vocabs(src_code, tgt_code, file_pair):
    """
    Creates the vocabs for each of the languages based on the training
    corpus.
    """
    src_vocab = Vocab(src_code)
    tgt_vocab = Vocab(tgt_code)

    src = read_file(file_pair[src_code])
    tgt = read_file(file_pair[tgt_code])

    for pair in zip(src, tgt):
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info(f'{src_vocab.lang_code} src vocab size: {src_vocab.n_words}')
    logging.info(f'{tgt_vocab.lang_code} tgt vocab size: {tgt_vocab.n_words}')

    return src_vocab, tgt_vocab


if __name__ == '__main__':
    # USEAGE
    src_vocab, tgt_vocab = make_vocabs('Original', 'Modern',
                                       {'Original':ORIGINAL_PATH,
                                        'Modern':MODERN_PATH})

    print(src_vocab.lang_code, src_vocab.n_words)
    print(tgt_vocab.lang_code, tgt_vocab.n_words)

    # 0 : 21079
    print(src_vocab.word2index[SOS_TOKEN], src_vocab.word2count[SOS_TOKEN])
    # 1 : 21079
    print(src_vocab.word2index[EOS_TOKEN], src_vocab.word2count[EOS_TOKEN])
