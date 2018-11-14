import torch
from torch.nn import Dropout, Embedding, Linear, Module, GRU, RNN
from torch.nn.functional import relu, log_softmax, softmax

from attention_models import build_attention_model


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
        attention = build_attention_model(args, args.hidden_size)

        device = torch.device('cpu')

        if args.decoder_mode == 'rnn':
            return DecoderRNN(args.hidden_size, output_size, device,
                              embedding_dropout=args.embedding_dropout,
                              lstm_dropout=args.lstm_dropout,
                              num_layers=args.decoder_layers,
                              attention=attention)
        elif args.decoder_mode == 'gru':
            return DecoderGRU(args.hidden_size, output_size, device,
                              embedding_dropout=args.embedding_dropout,
                              lstm_dropout=args.lstm_dropout,
                              num_layers=args.decoder_layers,
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
        if num_layers == 1:
            lstm_dropout = 0
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding_dropout = embedding_dropout
        self.lstm_dropout = lstm_dropout
        self.num_layers = num_layers
        self.attention = attention
        # Define layers below, aka embedding + RNN
        self.word_embedding = Embedding(output_size, hidden_size)
        self.dropout = Dropout(embedding_dropout)
        self.attention = attention
        self.rnn = RNN(hidden_size, hidden_size, num_layers=num_layers,
                       dropout=lstm_dropout, batch_first=True)
        self.linear = Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        """
        Runs the forward pass of the decoder.
        Returns the log_softmax, hidden state, and attn_weights.
        """
        # input tensor -> size (N, B, input_size)
        # hidden -> depends on RNN, see docs
        # use asserts to make sure correct sizes!
        # check if attention is NONE, if not do mode, else bypass
        # Apply embedding with dropout
        embedding = self.dropout(self.word_embedding(input))

        # Compute attention weights and apply them
        attended = None
        if self.attention is not None:
            attended, _ = self.attention(embedding, hidden, encoder_output)
        else:
            pass
            # ????????

        # Apply non-linear then RNN with hidden from encoder (later, decoder)
        output, hidden = self.rnn(attended, hidden)

        # Use softmax to pick most likely translation word embeddings
        output = log_softmax(self.linear(output), dim=2)

        return output, hidden


class DecoderGRU(Module):
    """A simple GRU decoder."""

    def __init__(self, hidden_size, output_size, device,
                 embedding_dropout=0.1, lstm_dropout=0.1, num_layers=1,
                 attention=None):
        """Initialize a word embedding and simple GRU decoder."""
        super().__init__()
        if num_layers == 1:
            lstm_dropout = 0
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding_dropout = embedding_dropout
        self.lstm_dropout = lstm_dropout
        self.num_layers = num_layers
        self.attention = attention
        # Define layers below, aka embedding + GRU
        self.word_embedding = Embedding(output_size, hidden_size)
        self.dropout = Dropout(embedding_dropout)
        self.attention = attention
        self.gru = GRU(hidden_size, hidden_size, num_layers=num_layers,
                       dropout=lstm_dropout, batch_first=True)
        self.linear = Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        """
        Runs the forward pass of the decoder.
        Returns the log_softmax, hidden state, and attn_weights.
        """
        # input tensor -> size (N, B, input_size)
        # hidden -> depends on RNN, see docs
        # use asserts to make sure correct sizes!
        # check if attention is NONE, if not do mode, else bypass
        # NOTE: handle when input is none ot generate from itself
        # Apply embedding with dropout
        embedding = self.dropout(self.word_embedding(input))

        # Compute attention weights and apply them
        attended = None
        if self.attention is not None:
            attended, _ = self.attention(embedding, hidden, encoder_output)
        else:
            # ???????
            pass
        # Apply non-linear then GRU with hidden from encoder (later, decoder)
        # print('DECODER')
        # print('A|H', attended.size(), hidden)
        output, hidden = self.gru(attended, hidden)
        # print('\n')
        # Use softmax to pick most likely translation word embeddings
        output = log_softmax(self.linear(output), dim=2)

        return output, hidden



if __name__ == '__main__':
    # params, confgurable

    from args import get_args
    from dataset import Dataset

    args = get_args()

    dataset = Dataset.load_from_args(args)

    decoder = build_decoder(args, dataset.vocab)

    args.decoder_mode = 'gru'
    decoder = build_decoder(args, dataset.vocab)

    args.decoder_mode = 'lol'
    try:
        decoder = build_decoder(args, dataset.vocab)
    except ValueError:
        print('Exception Thrown')
