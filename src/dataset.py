from itertools import accumulate, chain, groupby
import random
import sys
import torch
from utils import chunk, get_default_device

from vocab import Bivocabulary

TRAIN_PATH='../data/split/train.snt.aligned'
DEV_PATH='../data/split/dev.snt.aligned'
TEST_PATH='../data/split/test.snt.aligned'

# TODO: test
class Dataset:
    @classmethod
    def load_from_args(cls, args):
        vocab = Bivocabulary.create(args.source_language, args.target_language,
                                    args.reverse)

        train_sentences = args.train_file

        train_pairs = (vocab.add_sentence_pair(pair)
                       for pair in train_sentences)
        dev_pairs = (vocab.add_sentence_pair(pair)
                     for pair in args.dev_file)

        index = 0
        if args.reverse:
            index = 1

        device = get_default_device()

        test_sources = (vocab.source.add_sentence(sentence.split('|||', 1)[index])
                        for sentence in args.test_file)
        test_tensors = tuple(torch.tensor(s, dtype=torch.long,
                                          device=device).unsqueeze(0)
                             for s in test_sources)

        train_batches = _batch_by_source(train_pairs,
                                         batch_size=args.train_batch_size)
        dev_batches = _batch_by_source(dev_pairs)

        return cls(vocab, train_batches, dev_batches, test_tensors)

    def __init__(self, vocab, train_batches, dev_batches, test_sources):
        self.vocab = vocab
        self._train_batches = train_batches
        self._dev_batches = dev_batches
        self._dev_batch_cum_weights = \
            list(accumulate(len(batch) for batch in dev_batches))
        self._test_sources = test_sources

    def training_batches(self):
        # TODO: seed?
        return random.sample(self._train_batches, len(self._train_batches))

    def dev_batches(self):
        return self._dev_batches

    def random_dev_pairs(self, n=2):
        # TODO: seed?
        for _ in range(n):
            sources, targets, _ = \
                random.choices(self._dev_batches,
                               cum_weights=self._dev_batch_cum_weights)[0]
            i = random.choice(range(len(sources)))
            yield sources[i], targets[i]

    def test_sources(self):
        return self._test_sources

    def sentence_pair_from(self, source, target):
        for source_tensor, target_tensor in zip(source, target):
            source_sentence = self.vocab.source.sentence_from(source_tensor)
            target_sentence = self.vocab.target.sentence_from(target_tensor)

            yield source_sentence, target_sentence

    def info(self):
        return f'{len(self.vocab.source)} source words, ' \
               f'{len(self.vocab.target)} target words, ' \
               f'{len(self._train_batches)} train batches, ' \
               f'{len(self._dev_batches)} dev batches, ' \
               f'{len(self._test_sources)} test sentences'


def _batch_by_source(pairs, *, batch_size=None):
    by_source_length = lambda pair: len(pair[0])
    pairs = sorted(pairs, key=by_source_length)

    return tuple(_make_minibatch_pair(pairs)
                 for _, batch in groupby(pairs, key=by_source_length)
                 for pairs in chunk(batch, batch_size))


def _make_minibatch_pair(pairs):
    device = get_default_device()

    sources, targets = map(tuple, zip(*pairs))

    target_lens = torch.tensor([len(target) for target in targets],
                               device=device)
    pad_len = target_lens.max().item()
    targets_padded = tuple((target + (0,) * pad_len)[:pad_len]
                           for target in targets)

    source_minibatch = torch.tensor(sources, dtype=torch.long, device=device)
    target_minibatch = torch.tensor(targets_padded, dtype=torch.long,
                                    device=device)

    return source_minibatch, target_minibatch, target_lens




# EOF
