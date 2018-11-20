from argparse import ArgumentParser, FileType
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu

TEST_PATH='../../data/split/test.snt.aligned'


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--test-file',
                        type=FileType('r', encoding='utf-8', errors='ignore'),
                        default=TEST_PATH,
                        help='File containing test sentences')
    parser.add_argument('--eval-file',
                        type=FileType('r', encoding='utf-8', errors='ignore'),
                        required=True,
                        help='File containing model results for test file')
    parser.add_argument('--reverse', action='store_const',
                        default='False', const='True',
                        help='if True reverses source and target reading')

    args = parser.parse_args()
    args.reverse = True if args.reverse == 'True' else False
    return args


def create_triples(args):
    source_sentences, expected_sentences, actual_sentences = \
                                                    ([] for _ in range(3))
    for sentence in args.test_file:
        source, expected = tuple(sentence.rstrip().split('|||', 1))
        if args.reverse:
            expected_sentences.append(source)
            source_sentences.append(expected)
        else:
            expected_sentences.append(expected)
            source_sentences.append(source)

    actual_sentences = [sentence.strip() for sentence in args.eval_file]
    return source_sentences, expected_sentences, actual_sentences


def compare(source, expected, actual):
    for src, exp, act in zip(source, expected, actual):
        print(f'> {src}\n= {exp}\n< {act}\n')


def corpus_bleu(references, candidates):
    return nltk_corpus_bleu([[sentence.split()] for sentence in references],
                            [sentence.split() for sentence in candidates])


def main():
    args = get_args()

    source, expected, actual = create_triples(args)

    compare(source, expected, actual)

    bleu = corpus_bleu(expected, actual)
    print(f'Test BLEU: {bleu:.4f}\n')


if __name__ == '__main__':
    main()
