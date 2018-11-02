import torch
from vocab import SOS_INDEX, EOS_INDEX

"""File for handling translations"""

def tensor_from_sentence(vocab, sentence, device):
    """Returns a tensor for a raw sentence."""
    # Vocab(), str, device

    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass

    indexes.append(EOS_INDEX)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(source_vocab, target_vocab, pair, device):
    """Returns tensors for a raw sentence pair."""
    # Vocab(), Vocab(), (str, str), device

    input, target = pair

    input_tensor = tensor_from_sentence(source_vocab, input, device)
    target_tensor = tensor_from_sentence(target_vocab, target, device)

    return input_tensor, target_tensor
