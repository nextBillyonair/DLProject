import torch
from torch.nn import Dropout, Embedding, Linear, Module, Bilinear
from torch.nn.functional import relu, log_softmax, softmax

"""File for handiling Attention and their construction/init"""

def build_attention_model(args, hidden_size, bidirectional_encoder):
    """Builds the attention model to params."""

    if args.attention_mode == 'none':
        return None
    elif args.attention_mode == 'general':
        return GeneralAttention(hidden_size, bidirectional_encoder)
    elif args.attention_mode == 'concat':
        return ConcatAttention(hidden_size, bidirectional_encoder)
    else:
        raise ValueError('Invalid attention mode: %s' % (args.attention_mode))

# CONCAT
class ConcatAttention(Module):
    def __init__(self, hidden_size, bidirectional_encoder):
        super().__init__()
        self.hidden_size = hidden_size
        self.v_a = Linear(hidden_size, 1)

        if bidirectional_encoder:
            self.W_a = Linear(hidden_size * 3, hidden_size)
        else:
            self.W_a = Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, attendable):
        # Weights are learned from hidden state and encoder_outputs
        batch_size = attendable.size(0)
        sentence_length = attendable.size(1)

        hidden = hidden.view(batch_size, 1, self.hidden_size)
        hidden = hidden.expand(batch_size, sentence_length, self.hidden_size)

        hidden_with_attendable = torch.cat((hidden, attendable), dim=2)

        scores = torch.tanh(self.W_a(hidden_with_attendable))
        scores = self.v_a(scores)

        weights = softmax(scores, dim=1)
        weights = weights.view(batch_size, 1, sentence_length)

        # Apply weights
        attended = torch.bmm(weights, attendable)

        return attended, weights

# DOT
class GeneralAttention(Module):
    def __init__(self, hidden_size, bidirectional_encoder):
        super().__init__()
        self.hidden_size = hidden_size

        self.multiplier = 1
        if bidirectional_encoder:
            self.multiplier = 2

        self.W_a = Bilinear(self.hidden_size,
                            self.multiplier * self.hidden_size,
                            1)

    def forward(self, hidden, attendable):
        batch_size = attendable.size(0)
        sentence_length = attendable.size(1)

        hidden = hidden.view(batch_size, 1, self.hidden_size)
        hidden = hidden.expand(batch_size, sentence_length, self.hidden_size)

        hidden = hidden.contiguous()
        attendable = attendable.contiguous()

        scores = self.W_a(hidden, attendable)

        weights = softmax(scores, dim=1)
        weights = weights.view(batch_size, 1, sentence_length)

        # Apply weights
        attended = torch.bmm(weights, attendable)

        return attended, weights
