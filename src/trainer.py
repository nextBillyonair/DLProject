import torch
from loss import MaskedNLLLoss
from torch.optim import Adam
import random
import sys

from utils import corpus_bleu, Timer

class Trainer:
    @classmethod
    def create_from_args(cls, args, model, dataset):
        if args.load_checkpoint:
            trainer = torch.load(args.load_checkpoint)

            trainer._dataset = dataset
            trainer._num_epochs = args.epochs
            trainer._checkpoint_every = args.checkpoint_every
            trainer._checkpoint_prefix = args.checkpoint_prefix
            trainer._log_every = args.log_every
            trainer._teacher_forcing_chance = args.teacher_forcing_chance

            return trainer

        optimizer = Adam(model.parameters(), lr=args.initial_learning_rate)
        criterion = MaskedNLLLoss()

        return cls(model, dataset, optimizer, criterion,
                   num_epochs=args.epochs,
                   checkpoint_every=args.checkpoint_every,
                   checkpoint_prefix=args.checkpoint_prefix,
                   log_every=args.log_every,
                   teacher_forcing_chance=args.teacher_forcing_chance)


    def __getstate__(self):
        return self.model, self._optimizer, self._criterion, self._epoch, \
            self._total_loss


    def __setstate__(self, state):
        self.model, self._optimizer, self._criterion, self._epoch, \
            self._total_loss = state


    def __init__(self, model, dataset, optimizer, criterion, *, num_epochs,
                 checkpoint_every, checkpoint_prefix, log_every,
                 teacher_forcing_chance):
        self.model = model
        self._dataset = dataset
        self._optimizer = optimizer
        self._criterion = criterion

        self._epoch = 0
        self._num_epochs = num_epochs
        self._checkpoint_every = checkpoint_every
        self._checkpoint_prefix = checkpoint_prefix
        self._total_loss = 0
        self._log_every = log_every
        self._teacher_forcing_chance = teacher_forcing_chance

    def train(self):
        self._timer = Timer()

        for self._epoch in range(self._epoch + 1, self._num_epochs + 1):
            count = 0
            for batch in self._dataset.training_batches():
                loss = \
                    self.model.train(*batch, self._optimizer, self._criterion,
                                     self._teacher_forcing_chance)
                self._total_loss += loss
                print(count)
                count+=1

            if self._epoch % self._log_every == 0:
                self._log()

            if self._epoch % self._checkpoint_every == 0:
                self._checkpoint()

    def _checkpoint(self):
        path = f'{self._checkpoint_prefix}{self._epoch:011d}'

        print(f'\tsaving checkpoint {path}\n', file=sys.stderr)
        torch.save(self, path)

    def _log(self):
        percent = self._epoch / self._num_epochs * 100
        epochs = f'{self._epoch} / {self._num_epochs} ({percent:.2f}%)'
        progress = f'{self._timer.elapsed():.4f}s: {epochs}'
        avg_loss = self._total_loss / self._log_every
        print(f'[{progress}] avg loss: {avg_loss:.4f}\n', file=sys.stderr)

        for source, expected, actual in self._random_dev_translations():
            print(f'\t> {source}\n\t= {expected}\n\t< {actual}\n',
                  file=sys.stderr)

        bleu = corpus_bleu(*zip(*self._dev_translation_pairs()))
        print(f'\tDev BLEU: {bleu:.4f}\n', file=sys.stderr)

        self._total_loss = 0

    def _random_dev_translations(self, n=2):
        for source, target in self._dataset.random_dev_pairs(n):
            translation = self.model.translate(source.unsqueeze(0))[0]

            yield self._vocab.source.sentence_from(source), \
                self._vocab.target.sentence_from(target), \
                self._vocab.target.sentence_from(translation)

    def _dev_translation_pairs(self):
        """Yields (expected_sentence, actual_translation) pairs from dev."""
        for source, target, _ in self._dataset.dev_batches():
            translations = self.model.translate(source)

            expected = self._vocab.target.sentences_from(target)
            actual = self._vocab.target.sentences_from(translations)

            yield from zip(expected, actual)

    @property
    def _vocab(self):
        return self._dataset.vocab
