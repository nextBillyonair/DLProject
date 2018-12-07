from argparse import ArgumentParser, FileType

TEST_PATH='../../data/split/test.snt.aligned'
FILES=['rnn.txt', 'gru.txt', 'bigru.txt', 'bigru_general.txt',
       'bigru_general_50.txt', 'bigru_general_100.txt', 'bigru_concat.txt',
       'bigru_concat_50.txt', 'bigru_concat_100.txt', 'best_file.txt']


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--test-file',
                        type=FileType('r', encoding='utf-8', errors='ignore'),
                        default=TEST_PATH,
                        help='File containing test sentences')
    parser.add_argument('--reverse', action='store_const',
                        default='False', const='True',
                        help='if True reverses source and target reading')


    args = parser.parse_args()
    args.reverse = True if args.reverse == 'True' else False
    return args


def read_data(args):
    pairings = {'source': [], 'target':[]}

    for sentence in args.test_file:
        source = expected = None
        if args.reverse:
            expected, source = tuple(sentence.strip().split('|||', 1))
        else:
            source, expected = tuple(sentence.strip().split('|||', 1))
        pairings['source'].append(source)
        pairings['target'].append(expected)

    for file_name in FILES:
        key = file_name.split('.')[0]
        with open(file_name, 'r') as input_file:
            sentences = [sentence
                         for sentence in input_file.read().split('\n')]
            pairings[key] = sentences

    return pairings


def zip_data(pairings):
    return zip(pairings['source'], pairings['rnn'], pairings['gru'],
               pairings['bigru'], pairings['bigru_general'],
               pairings['bigru_general_50'], pairings['bigru_general_100'],
               pairings['bigru_concat'], pairings['bigru_concat_50'],
               pairings['bigru_concat_100'], pairings['best_file'],
               pairings['target'])


def main():
    args = get_args()
    pairings = read_data(args)

    zipped_data = zip_data(pairings)

    model_names = ['source'] + \
                  [file_name.split('.')[0] for file_name in FILES] \
                  + ['target']

    for pairs in zipped_data:
        for index, result in enumerate(pairs):
            msg = f'{model_names[index].ljust(20)} > {result}'
            print(msg)
        print('\n')


if __name__ == '__main__':
    main()
