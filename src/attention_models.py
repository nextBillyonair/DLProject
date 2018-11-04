import torch
from torch.nn import Dropout, Embedding, Linear, Module
from torch.nn.functional import relu, log_softmax

"""
# Spin off to separate class, maybe don't include in this file
Attention Modes:
None
original?
general
dot
concat
"""

def build_attention_model(mode):
    """Builds the attention model to params."""
    pass
    choices=[None,'original','general', 'dot',
             'concat', 'location']
    if mode is None:
        return None
    elif mode is 'original':
        return None
    elif mode is 'general':
        return None
    elif mode is 'dot':
        return None
    elif mode is 'concat':
        return None
    elif mode is 'location':
        return None
    else:
        raise ValueError('Invalid attention mode: %s' % (mode))


class Attention(Module):
    def __init__(self, hidden_size, max_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length


    def forward(self, input, hidden, attendable):
        # Run forward of attention model
        return output, weights
