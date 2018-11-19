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
        bidirectional_encoder = True if args.encoder_mode == 'bigru' else False
        attention = build_attention_model(args, args.hidden_size,
                                          bidirectional_encoder)

        if args.decoder_mode == 'rnn':
            return DecoderRNN(args.hidden_size, output_size,
                              embedding_dropout=args.embedding_dropout,
                              rnn_dropout=args.rnn_dropout,
                              num_layers=args.decoder_layers,
                              attention=attention,
                              bidirectional_encoder=bidirectional_encoder)
        elif args.decoder_mode == 'gru':
            return DecoderGRU(args.hidden_size, output_size,
                              embedding_dropout=args.embedding_dropout,
                              rnn_dropout=args.rnn_dropout,
                              num_layers=args.decoder_layers,
                              attention=attention,
                              bidirectional_encoder=bidirectional_encoder)
        else:
            raise ValueError('Invalid decoder mode: %s' % (args.decoder_mode))


# TEMPLATE CLASS
class DecoderRNN(Module):
    """A simple RNN decoder."""

    def __init__(self, hidden_size, output_size,
                 embedding_dropout=0.1, rnn_dropout=0.1, num_layers=1,
                 attention=None, bidirectional_encoder=True):
        """Initialize a word embedding and simple RNN decoder."""
        super().__init__()
        if num_layers == 1:
            rnn_dropout = 0
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dropout = embedding_dropout
        self.rnn_dropout = rnn_dropout
        self.num_layers = num_layers
        self.attention = attention
        # Define layers below, aka embedding + RNN
        self.word_embedding = Embedding(output_size, hidden_size)
        self.dropout = Dropout(embedding_dropout)
        self.attention = attention
        self.multiplier = 1
        if attention is not None and bidirectional_encoder:
            self.multiplier = 3
        elif attention is not None and not bidirectional_encoder:
            self.multiplier = 2

        self.rnn = RNN(self.multiplier * hidden_size, hidden_size,
                       num_layers=num_layers,
                       dropout=rnn_dropout, batch_first=True)
        self.linear = Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        """
        Runs the forward pass of the decoder.
        """
        # Apply embedding with dropout
        embedding = self.dropout(self.word_embedding(input))

        # Compute attention weights and apply them
        attended = None
        if self.attention is not None:
            attended, _ = self.attention(hidden, encoder_output)
            attended = torch.cat((attended, embedding), 2) #2?
        else:
            attended = embedding

        # Apply non-linear then RNN with hidden from encoder (later, decoder)
        output, hidden = self.rnn(attended, hidden)

        # Use softmax to pick most likely translation word embeddings
        output = log_softmax(self.linear(output), dim=2)

        return output, hidden


class DecoderGRU(Module):
    """A simple GRU decoder."""

    def __init__(self, hidden_size, output_size,
                 embedding_dropout=0.1, rnn_dropout=0.1, num_layers=1,
                 attention=None, bidirectional_encoder=True):
        """Initialize a word embedding and simple GRU decoder."""
        super().__init__()
        if num_layers == 1:
            rnn_dropout = 0
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dropout = embedding_dropout
        self.rnn_dropout = rnn_dropout
        self.num_layers = num_layers
        self.attention = attention

        self.word_embedding = Embedding(output_size, hidden_size)
        self.dropout = Dropout(embedding_dropout)
        self.attention = attention

        self.multiplier = 1
        if attention is not None and bidirectional_encoder:
            self.multiplier = 3
        elif attention is not None and not bidirectional_encoder:
            self.multiplier = 2

        self.gru = GRU(self.multiplier*hidden_size, hidden_size, num_layers=num_layers,
                       dropout=rnn_dropout, batch_first=True)
        self.linear = Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        """
        Runs the forward pass of the decoder.
        """
        # Apply embedding with dropout
        embedding = self.dropout(self.word_embedding(input))

        # Compute attention weights and apply them
        attended = None
        if self.attention is not None:
            attended, _ = self.attention(hidden, encoder_output)
            attended = torch.cat((attended, embedding), 2)
        else:
            attended = embedding

        # Apply non-linear then GRU with hidden from encoder (later, decoder)
        output, hidden = self.gru(attended, hidden)

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
