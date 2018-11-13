from argparse import ArgumentParser
import torch

from vocab import TRAIN_PATH, DEV_PATH, TEST_PATH

def get_args():
    """Defines All CMD Line Arguments"""
    # Feel free to add arguments as needed
    parser = ArgumentParser()

    # Hidden Size
    parser.add_argument('--hidden-size', default=256, type=int,
                        help='hidden size of encoder/decoder (word vector '
                             'size)')

    # Hyperparams
    parser.add_argument('--initial-learning-rate', default=0.001, type=int,
                        help='initial learning rate')
    parser.add_argument('--embedding-dropout', default=0.1, type=float,
                        help='training embedding dropout rate')
    parser.add_argument('--lstm-dropout', default=0.1, type=float,
                        help='training LSTM dropout rate')
    parser.add_argument('--encoder-layers', default=1, type=int,
                        help='number of LSTM layers in the encoder')
    parser.add_argument('--decoder-layers', default=1, type=int,
                        help='number of LSTM layers in the decoder')
    parser.add_argument('--encoder-mode', default='baseline',
                        choices=['baseline','gru','bidirectional'],
                        help='type of encoder used')
    parser.add_argument('--decoder-mode', default='baseline',
                        choices=['baseline','gru'],
                        help='type of decoder used')
    parser.add_argument('--attention-mode', default=None,
                        choices=[None,'dot','concat'],
                        help='type of attention used')


    # DATASET ARGS
    parser.add_argument('--source-language', default='original',
                       help='source sentence language')
    parser.add_argument('--target-language', default='modern',
                       help='target sentence language')
    parser.add_argument('--train-file', type=FileType('r'),
                       default=TRAIN_PATH, help='train sentences')
    parser.add_argument('--dev-file', type=FileType('r'),
                       default=DEV_PATH, help='dev sentences')
    parser.add_argument('--test-file', type=FileType('r'),
                       default=TEST_PATH, help='test sentences')
    parser.add_argument('--train-batch-size', type=int, default=128)



    # DEVICE: CPU or GPU
    parser.add_argument('--device', type=torch.device, default='cpu',
                        help='use CPU/CUDA for training/evaluation')

    # TRAINING ARGS
    parser.add_argument('--epochs', default=500, type=int,
                       help='total number of epochs to train on')
    parser.add_argument('--log-every', default=10, type=int,
                       help='log loss info every this many epochs')
    parser.add_argument('--checkpoint-every', default=100, type=int,
                       help='write out checkpoint every this many epochs')
    parser.add_argument('--checkpoint-prefix', default='checkpoint_',
                       help='checkpoint filename prefix')
    parser.add_argument('--initial-learning-rate', default=0.001, type=int,
                       help='initial learning rate')
    parser.add_argument('--teacher-forcing-chance', default=0, type=float,
                       help='percent of batches on which to teacher force')

    parser.add_argument('--load-checkpoint', help='training checkpoint to load')


    # parse and return
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
