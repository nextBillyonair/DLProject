import torch
from torch.nn import Dropout, Embedding, Linear, Module, GRU, RNN
from torch.nn.functional import relu, log_softmax, softmax

from attention_models import build_attention_model


"""File for handiling Decoders and their construction/init"""
def build_decoder(args, vocab):
    """Builds the decoder to params."""

    output_size = len(vocab.target)

    bidirectional_encoder = True if args.encoder_mode == 'bigru' else False
    attention = build_attention_model(args, args.hidden_size,
                                      bidirectional_encoder)

    rnn = build_rnn_layer(args)
    return Decoder(args.hidden_size, output_size, rnn, attention,
                   args.rnn_dropout)

def build_rnn_layer(args):
    multiplier = 1
    if args.attention_mode != 'none' and args.encoder_mode == 'bigru':
        multiplier = 3
    elif args.attention_mode != 'none' and args.encoder_mode != 'bigru':
        multiplier = 2

    dropout = args.rnn_dropout if args.decoder_layers != 1 else 0

    if args.decoder_mode == 'rnn':
        return RNN(multiplier * args.hidden_size, args.hidden_size,
                   num_layers=args.decoder_layers,
                   dropout=dropout, batch_first=True)
    elif args.decoder_mode == 'gru':
        return GRU(multiplier * args.hidden_size, args.hidden_size,
                   num_layers=args.decoder_layers,
                   dropout=dropout, batch_first=True)
    else:
        raise ValueError('Invalid decoder mode: %s' % (args.decoder_mode))


class Decoder(Module):
    def __init__(self, hidden_size, output_size, rnn_layer, attention=None,
                 embedding_dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = rnn_layer
        self.attention = attention
        self.word_embedding = Embedding(output_size, hidden_size)
        self.dropout = Dropout(embedding_dropout)
        self.linear = Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        embedding = self.dropout(self.word_embedding(input))

        attended = None
        if self.attention is not None:
            attended, _ = self.attention(hidden, encoder_output)
            attended = torch.cat((attended, embedding), 2) #2?
        else:
            attended = embedding

        output, hidden = self.rnn(attended, hidden)
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
