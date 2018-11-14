from nltk.tokenize import word_tokenize
import spacy
import logging
from random import shuffle
from glob import glob

"""File for preprocessing data, and merging"""

# POS MODEL
NLP=spacy.load('en')

# DATA PATHS
MODERN_PATH='../../data/modern/'
ORIGINAL_PATH='../../data/original/'
PROCESSED_PATH='../../data/processed/'
SPLIT_PATH='../../data/split/'


def get_files(path=PROCESSED_PATH):
    """Given path, returns files in path directory"""
    return glob(path+'*')


def read_file(input_file):
    """Reads a file"""
    logging.info("Reading lines of %s...", input_file)
    return open(input_file).read().strip().split('\n')


def tokenize(raw_string):
    """Uses nltk to tokenize string"""
    return word_tokenize(raw_string.lower())


def replace_proper_nouns(raw_string):
    """Replaces all proper nouns with token propn"""
    doc = NLP(raw_string)
    return ' '.join([token.text if token.pos_ is not 'PROPN' else 'propn'
                     for token in doc])


def process_string(raw_string):
    """Process a string by tokenizing, and rejoining"""
    return ' '.join(tokenize(replace_proper_nouns(raw_string)))


def process_play(play):
    """Processes an entire play:list(str)"""
    return list(map(process_string, play))


def get_file_name(path):
    """Strips the path directory from path"""
    return path.split('/')[-1]


def split_data(data, train_p=0.875, dev_p=0.1):
    """Splits the data into train/dev/test"""
    shuffle(data)
    length = len(data)
    train_slice = int(train_p*length)
    dev_slice = int(dev_p*length)
    return data[:train_slice], \
           data[train_slice:train_slice+dev_slice], \
           data[train_slice+dev_slice:]


if __name__ == '__main__':

    # Modern Process
    modern_files = get_files(MODERN_PATH)
    for play in modern_files:
        print(play)
        processed = process_play(read_file(play))
        file_name = PROCESSED_PATH + 'modern/' + get_file_name(play)
        with open(file_name, 'w') as outfile:
            for sentence in processed:
                outfile.write("%s\n" % entence)

    # Original Process
    original_files = get_files(ORIGINAL_PATH)
    for play in original_files:
        print(play)
        processed = process_play(read_file(play))
        file_name = PROCESSED_PATH + 'original/' + get_file_name(play)
        with open(file_name, 'w') as outfile:
            for sentence in processed:
                outfile.write("%s\n" % sentence)

    # Merge Modern
    modern_files = sorted(get_files(PROCESSED_PATH + 'modern/'))
    file_name = PROCESSED_PATH + 'modern.snt.aligned'
    with open(file_name, 'w') as outfile:
        for play in modern_files:
            play_sentences = read_file(play)
            print(get_file_name(play), len(play_sentences))
            for sentence in play_sentences:
                outfile.write("%s\n" % sentence)

    # Merge Original
    original_files = sorted(get_files(PROCESSED_PATH + 'original/'))
    file_name = PROCESSED_PATH + 'original.snt.aligned'
    with open(file_name, 'w') as outfile:
        for play in original_files:
            play_sentences = read_file(play)
            print(get_file_name(play), len(play_sentences))
            for sentence in play_sentences:
                outfile.write("%s\n" % sentence)

    #Train/Dev/Test
    # 21079 21079
    # 18444 2107 528
    original = read_file(PROCESSED_PATH + 'original.snt.aligned')
    modern = read_file(PROCESSED_PATH + 'modern.snt.aligned')
    pairs = zip(original, modern)
    # do a better split via looking at vocab
    train, dev, test = split_data(list(pairs))

    print(len(original), len(modern))
    print(len(train), len(dev), len(test))

    file_name = SPLIT_PATH + 'train.snt.aligned'
    with open(file_name, 'w') as outfile:
        for pair in train:
            outfile.write("%s|||%s\n" % (pair[0], pair[1]))
    file_name = SPLIT_PATH + 'dev.snt.aligned'
    with open(file_name, 'w') as outfile:
        for pair in dev:
            outfile.write("%s|||%s\n" % (pair[0], pair[1]))
    file_name = SPLIT_PATH + 'test.snt.aligned'
    with open(file_name, 'w') as outfile:
        for pair in test:
            outfile.write("%s|||%s\n" % (pair[0], pair[1]))
