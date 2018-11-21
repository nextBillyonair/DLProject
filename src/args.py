from argparse import ArgumentParser, FileType
import torch

from utils import set_default_device, get_default_device

TRAIN_PATH='../data/split/train.snt.aligned'
DEV_PATH='../data/split/dev.snt.aligned'
TEST_PATH='../data/split/test.snt.aligned'

def get_args():
    """Defines All CMD Line Arguments"""
    # Feel free to add arguments as needed
    parser = ArgumentParser()

    # Hidden Size
    parser.add_argument('--hidden-size', default=256, type=int,
                        help='hidden size of encoder/decoder (word vector '
                             'size)')

    # Hyperparams
    parser.add_argument('--embedding-dropout', default=0.1, type=float,
                        help='training embedding dropout rate')
    parser.add_argument('--rnn-dropout', default=0.1, type=float,
                        help='training RNN/GRU dropout rate')
    parser.add_argument('--encoder-layers', default=1, type=int,
                        help='number of RNN layers in the encoder')
    parser.add_argument('--decoder-layers', default=1, type=int,
                        help='number of RNN layers in the decoder')
    parser.add_argument('--encoder-mode', default='bigru',
                        choices=['rnn','gru','bigru'],
                        help='type of encoder used')
    parser.add_argument('--decoder-mode', default='gru',
                        choices=['rnn','gru'],
                        help='type of decoder used')
    parser.add_argument('--attention-mode', default='concat',
                        choices=['none','general','concat'],
                        help='type of attention used')


    # DATASET ARGS
    parser.add_argument('--source-language', default='original',
                       help='source sentence language')
    parser.add_argument('--target-language', default='modern',
                       help='target sentence language')
    parser.add_argument('--train-file', type=FileType('r', encoding='utf-8'),
                       default=TRAIN_PATH, help='train sentences')
    parser.add_argument('--dev-file', type=FileType('r', encoding='utf-8'),
                       default=DEV_PATH, help='dev sentences')
    parser.add_argument('--test-file', type=FileType('r', encoding='utf-8'),
                       default=TEST_PATH, help='test sentences')
    parser.add_argument('--train-batch-size', type=int, default=128,
                        help='sets the training batch size')
    parser.add_argument('--reverse', action='store_const',
                        default='False', const='True',
                        help='if True reverses source and target reading')

    # MAX LEN
    parser.add_argument('--max-length', type=int, default=500,
                        help='max length of a decoded output sentence')

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

    # GPU USAGE
    parser.add_argument('--use-gpu', dest='device', action='store_const',
                        default='cpu', const='cuda',
                        help='use CPU/CUDA for training/evaluation')

    # parse and return
    args = parser.parse_args()

    # Set device for all tensors + models
    set_default_device(args.device)
    # Convert to bool
    args.reverse = True if args.reverse == 'True' else False
    # defualt to 1 b/c sizes are difficult to match
    args.encoder_layers = args.decoder_layers = 1

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
