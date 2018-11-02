from nltk.tokenize import word_tokenize
import spacy

# POS MODEL
NLP=spacy.load('en')

# DATA PATHS
MODERN_PATH='../../data/modern/'
ORIGINAL_PATH='../../data/original/'
PROCESSED_PATH='../../data/processed/'

def get_files(path=PROCESSED_PATH):
    return glob(path+'*')

def read_file(input_file):
    logging.info("Reading lines of %s...", input_file)
    return open(input_file).read().strip().split('\n')

def tokenize(raw_string):
    return word_tokenize(raw_string.lower())

def replace_proper_nouns(raw_string):
    doc = NLP(raw_string)
    return ' '.join([token.text if token.pos_ is not 'PROPN' else 'propn'
                     for token in doc])

def process_string(raw_string):
    return ' '.join(tokenize(replace_proper_nouns(raw_string)))

def process_play(play):
    return list(map(process_string, play))

def get_file_name(path):
    return path.split('/')[-1]


if __name__ == '__main__':

    # Modern Process
    modern_files = get_files(MODERN_PATH)
    for play in modern_files:
        print(play)
        processed = process_play(read_file(play))
        file_name = PROCESSED_PATH + 'modern/' + get_file_name(play)
        with open(file_name, 'w') as outfile:
            for sentence in processed:
                new_sentence = "<SOS> " + sentence + " <EOS>"
                outfile.write("%s\n" % new_sentence)

    # Original Process
    original_files = get_files(ORIGINAL_PATH)
    for play in original_files:
        print(play)
        processed = process_play(read_file(play))
        file_name = PROCESSED_PATH + 'original/' + get_file_name(play)
        with open(file_name, 'w') as outfile:
            for sentence in processed:
                new_sentence = "<SOS> " + sentence + " <EOS>"
                outfile.write("%s\n" % new_sentence)

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
