import torch
from torch.nn import Dropout, Embedding, Linear, Module, GRU
from torch.nn.functional import relu, log_softmax, softmax

"""File for handiling Encoders and their construction/init"""

"""
LIST MODE CONVENTIONS HERE:
baseline -> simple RNN (or should this be RNN)
gru -> GRU
bidirectional -> bidirectional GRU
"""
# change to args
def build_encoder(args, vocab):
        """Builds the encoder to params."""
        device = args.device
        input_size = len(vocab.source)

        device = torch.device('cpu')

        if args.encoder_mode is 'baseline':
            return EncoderRNN(input_size, args.hidden_size, device,
                              dropout=args.lstm_dropout,
                              num_layers=args.num_layers)
        elif args.encoder_mode is 'gru':
            return EncoderGRU(input_size, args.hidden_size, device,
                              dropout=args.lstm_dropout,
                              num_layers=args.num_layers)
        elif args.encoder_mode is 'bidirectional':
            return EncoderBidirectional(input_size, args.hidden_size, device,
                                        dropout=args.lstm_dropout,
                                        num_layers=args.num_layers)
        else:
            raise ValueError('Invalid encoder mode: %s' % (args.encoder_mode))

# ENCODER TEMPLATE, FOLLOW METHODS
# NAME CONVENTION: Encoder{ExtentionName}
class EncoderRNN(Module):
    """A word embedding, simple RNN encoder."""

    def __init__(self, input_size, hidden_size, device,
                 dropout=0.1, num_layers=1):
        """Initialize a word embedding and simple RNN encoder."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = dropout
        self.num_layers = num_layers
        # Define layers below, aka embedding + RNN

    def forward(self, input, hidden=None):
        """
        Runs the forward pass of the encoder returning the output and the
        hidden state.
        """
        # input tensor -> size (N, B, input_size)
        # hidden -> depends on RNN, see docs
        # use asserts to make sure correct sizes!
        return input, hidden


class EncoderGRU(Module):
    """A word embedding, GRU encoder."""

    def __init__(self, input_size, hidden_size, device,
                 dropout=0.1, num_layers=1):
        """Initialize a word embedding and GRU encoder."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = dropout
        self.num_layers = num_layers
        # Define layers below, aka embedding + GRU

    def forward(self, input, hidden=None):
        """
        Runs the forward pass of the encoder returning the output and the
        hidden state.
        """
        # input tensor -> size (N, B, input_size)
        # hidden -> depends on RNN, see docs
        # use asserts to make sure correct sizes!
        return input, hidden


class EncoderBidirectional(Module):
    """A word embedding and bi-directional GRU encoder."""

    def __init__(self, input_size, hidden_size, device,
                 dropout=0.1, num_layers=1):
        """Initialize a word embedding and bi-directional GRU encoder."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = dropout
        self.num_layers = num_layers
        # Define layers below, aka embedding + BiGRU

    def forward(self, input, hidden=None):
        """
        Runs the forward pass of the encoder returning the output and the
        hidden state.
        """
        # input tensor -> size (N, B, input_size)
        # hidden -> depends on RNN, see docs
        # use asserts to make sure correct sizes!
        return input, hidden




if __name__ == '__main__':
    # params, confgurable
    input_size = 10
    hidden_size = 1000
    hidden = None
    input_tensor = None

    encoder = build_encoder(input_size, hidden_size)
    # test forward
    encoder = build_encoder(input_size, hidden_size, mode='gru')
    # test forward
    encoder = build_encoder(input_size, hidden_size, mode='bidirectional')
    # test forward

    try:
        encoder = build_encoder(input_size, hidden_size, mode='lol')
    except ValueError:
        pass
