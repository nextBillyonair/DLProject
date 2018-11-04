import torch
from vocab import SOS_INDEX, EOS_INDEX, UNK_INDEX

"""File for handling conversions of str -> tensor"""

def tensor_from_sentence(vocab, sentence, device):
    """Returns a tensor for a raw sentence."""
    # Vocab(), str, device

    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            # pass
            indexes.append(UNK_INDEX)

    # indexes.append(EOS_INDEX) # uncomment if no eos token
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(source_vocab, target_vocab, pair, device):
    """Returns tensors for a raw sentence pair."""
    # Vocab(), Vocab(), (str, str), device

    input, target = pair

    input_tensor = tensor_from_sentence(source_vocab, input, device)
    target_tensor = tensor_from_sentence(target_vocab, target, device)

    return input_tensor, target_tensor


if __name__ == '__main__':
    # usage
    from vocab import make_vocabs
    src_vocab, tgt_vocab = make_vocabs('Original', 'Modern')

    # Random set from test + fake word to test unk
    src = "<SOS> sit down and feed , and welcome to our table bbbbbbb . <EOS>"
    tgt = "<SOS> sit down and eat , and welcome to our table bbbbbbb . <EOS>"

    srs_t, tgt_t = tensors_from_pair(src_vocab,
                                     tgt_vocab,
                                     (src, tgt),
                                     torch.device('cpu'))
    print(srs_t)
    print(tgt_t)
