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
    parser.add_argument('--stats-only', action='store_const',
                        default='False', const='True',
                        help='if True only outputs bleu')


    args = parser.parse_args()
    args.reverse = True if args.reverse == 'True' else False
    args.stats_only = True if args.stats_only == 'True' else False
    return args


def create_triples(args):
    source_sentences, expected_sentences, actual_sentences = \
                                                    ([] for _ in range(3))
    for sentence in args.test_file:
        source, expected = tuple(sentence.strip().split('|||', 1))
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

    if not args.stats_only:
        compare(source, expected, actual)

    # take tgt bleu
    tgt_bleu = corpus_bleu(expected, actual)
    print(f'Test TGT BLEU: {tgt_bleu:.4f}')
    src_bleu = corpus_bleu(source, actual)
    print(f'Test SRC BLEU: {src_bleu:.4f}')


if __name__ == '__main__':
    main()
