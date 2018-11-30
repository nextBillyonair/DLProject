import torch
from torch.nn import Dropout, Embedding, Linear, Module, GRU, RNN
from torch.nn.functional import relu, log_softmax, softmax

"""File for handiling Encoders and their construction/init"""

def build_encoder(args, vocab):
    """Builds the encoder to params."""

    input_size = len(vocab.source)
    rnn_layer = None
    bidirectional = False if args.encoder_mode != 'bigru' else True
    dropout = args.rnn_dropout if args.encoder_layers != 1 else 0

    if args.encoder_mode == 'rnn':
        rnn_layer = RNN(args.hidden_size, args.hidden_size,
                        num_layers=args.encoder_layers, dropout=dropout,
                        batch_first=True)
    elif args.encoder_mode == 'gru' or args.encoder_mode == 'bigru':
        rnn_layer = GRU(args.hidden_size, args.hidden_size,
                        num_layers=args.encoder_layers, dropout=dropout,
                        bidirectional=bidirectional, batch_first=True)
    else:
        raise ValueError('Invalid encoder mode: %s' % (args.encoder_mode))

    return Encoder(input_size, args.hidden_size, rnn_layer,
                   bidirectional=bidirectional)


class Encoder(Module):
    def __init__(self, input_size, hidden_size, rnn_layer,
                 bidirectional=False):
        """Initialize a word embedding and simple RNN encoder."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn_layer = rnn_layer
        self.word_embedding = Embedding(input_size, hidden_size)

    def forward(self, input, hidden=None):
        """
        Runs the forward pass of the encoder; returns (output, hidden state).
        """
        output, hidden = self.rnn_layer(self.word_embedding(input), hidden)
        if self.bidirectional: return output, hidden[:len(hidden)//2]
        return output, hidden



if __name__ == '__main__':
    # params, confgurable
    from args import get_args
    from dataset import Dataset

    args = get_args()

    dataset = Dataset.load_from_args(args)

    encoder = build_encoder(args, dataset.vocab)

    args.encoder_mode = 'gru'
    encoder = build_encoder(args, dataset.vocab)

    args.encoder_mode = 'bidirectional'
    encoder = build_encoder(args, dataset.vocab)

    args.encoder_mode = 'lol'
    try:
        encoder = build_encoder(args, dataset.vocab)
    except ValueError:
        print('Exception Thrown')
