import torch
from torch.nn import Dropout, Embedding, Linear, Module, GRU
from torch.nn.functional import relu, log_softmax, softmax


"""File for handiling Decoders and their construction/init"""

"""
LIST MODE CONVENTIONS HERE:
RNN Layer:
baseline -> simple RNN (or should this be RNN)
gru -> GRU
NOTE: Decoders do not do bidirectional
"""
# change to args
def build_decoder(args, vocab):
        """Builds the decoder to params."""

        output_size = len(vocab.target)

        # HANDLE ATTENTION HERE

        device = torch.device('cpu')

        if args.decoder_mode is 'baseline':
            return DecoderRNN(args.hidden_size, output_size, device,
                              embedding_dropout=args.embedding_dropout,
                              lstm_dropout=args.lstm_dropout,
                              num_layers=args.num_layers,
                              attention=attention)
        elif args.decoder_mode is 'gru':
            return DecoderGRU(args.hidden_size, output_size, device,
                              embedding_dropout=args.embedding_dropout,
                              lstm_dropout=args.lstm_dropout,
                              num_layers=args.num_layers,
                              attention=attention)
        else:
            raise ValueError('Invalid decoder mode: %s' % (args.decoder_mode))


# TEMPLATE CLASS
class DecoderRNN(Module):
    """A simple RNN decoder."""

    def __init__(self, hidden_size, output_size, device,
                 embedding_dropout=0.1, lstm_dropout=0.1, num_layers=1,
                 attention=None):
        """Initialize a word embedding and simple RNN decoder."""
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding_dropout = embedding_dropout
        self.lstm_dropout = lstm_dropout
        self.num_layers = num_layers
        self.attention = attention
        # Define layers below, aka embedding + RNN

    def forward(self, encoder_outputs, input=None, hidden=None):
        """
        Runs the forward pass of the decoder.
        Returns the log_softmax, hidden state, and attn_weights.
        """
        # input tensor -> size (N, B, input_size)
        # hidden -> depends on RNN, see docs
        # use asserts to make sure correct sizes!
        # check if attention is NONE, if not do mode, else bypass
        return input, hidden


class DecoderGRU(Module):
    """A simple GRU decoder."""

    def __init__(self, hidden_size, output_size, device,
                 embedding_dropout=0.1, lstm_dropout=0.1, num_layers=1,
                 attention=None):
        """Initialize a word embedding and simple GRU decoder."""
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding_dropout = embedding_dropout
        self.lstm_dropout = lstm_dropout
        self.num_layers = num_layers
        self.attention = attention
        # Define layers below, aka embedding + RNN

    def forward(self, encoder_outputs, input=None, hidden=None):
        """
        Runs the forward pass of the decoder.
        Returns the log_softmax, hidden state, and attn_weights.
        """
        # input tensor -> size (N, B, input_size)
        # hidden -> depends on RNN, see docs
        # use asserts to make sure correct sizes!
        # check if attention is NONE, if not do mode, else bypass
        # NOTE: handle when input is none ot generate from itself
        return input, hidden



if __name__ == '__main__':
    # params, confgurable
    hidden_size = 10
    output_size = 1000

    decoder = build_decoder(hidden_size, output_size)
    # test forward
    decoder = build_decoder(hidden_size, output_size, mode='gru')
    # test forward

    try:
        de = build_decoder(hidden_size, output_size, mode='lol')
    except ValueError:
        pass
