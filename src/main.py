import torch
import sys

from args import get_args
from dataset import Dataset
from model import Model
from trainer import Trainer

def main():
    args = get_args()

    dataset = Dataset.load_from_args(args)
    print(f'Loaded dataset: {dataset.info()}', file=sys.stderr)

    model = Model.create_from_args(args, dataset.vocab, args.max_length)
    trainer = Trainer.create_from_args(args, model, dataset)

    # Train model
    print('Training model...', file=sys.stderr)
    trainer.train()

    # Output translations of test set
    print('Translating test set...', file=sys.stderr)
    for source_sentence in dataset.test_sources():
        translation = trainer.model.translate(source_sentence)[0]
        print(dataset.vocab.target.sentence_from(translation))


if __name__ == '__main__':
    main()
