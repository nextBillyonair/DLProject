import torch
from torch.nn import Dropout, Embedding, Linear, Module
from torch.nn.functional import relu, log_softmax

"""
# Spin off to separate class, maybe don't include in this file
Attention Modes:
dot
concat
None - identity
"""

# FIX ARGS
def build_attention_model(args, hidden_size):
    """Builds the attention model to params."""

    if args.attention_mode is None:
        return None
    elif args.attention_mode == 'dot':
        return None
    elif args.attention_mode == 'concat':
        return ConcatAttention(hidden_size, args.max_length)
    else:
        raise ValueError('Invalid attention mode: %s' % (args.attention_mode))


# # IDENTITY
# class Attention(Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, *args):
#         return args


# CONCAT
class ConcatAttention(Module):
    def __init__(self, hidden_size, max_length):
        super().__init__()

        self.get_weights = Linear(2*hidden_size, max_length)
        self.pay_attention = Linear(hidden_size * 3, hidden_size)

    def forward(self, input, hidden, attendable):
        # Weights are learned from input and hidden state
        # print('\n')
        # print('ATTENTION DEBUG:')
        # print('I|H|A', input.size(), hidden[0].size(), attendable.size())
        hidden = hidden[0].view(*input.size())
        input_with_hidden = torch.cat((input, hidden), 2)

        sentence_length = attendable.size(1)
        # print('IWH|SL', input_with_hidden.size(), sentence_length)
        weights = log_softmax(self.get_weights(input_with_hidden),
                              dim=2)[:, :, :sentence_length]
        # print('W|A', weights.size(), attendable.size())
        # Apply weights
        with_attention = torch.bmm(weights, attendable)

        # Apply attention to input
        input_with_attention = torch.cat((input, with_attention), 2)
        # print('IWA', input_with_attention.size())
        attended = self.pay_attention(input_with_attention)
        # print('DONE\n')

        return relu(attended), weights


# DOT
class DotAttention(Module):
    def __init__(self, hidden_size, max_length):
        super().__init__()

    def forward(self, input, hidden, attendable):
        return None
