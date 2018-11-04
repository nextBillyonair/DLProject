import random

from translate import translate


def translate_sentences(encoder, decoder, pairs, source_vocab, target_vocab,
                        device, max_num_sentences=None):
    """Translate dev/test sets and returns their translations."""
    output_sentences = []

    for source, _ in pairs[:max_num_sentences]:
        output_words, _ = translate(encoder, decoder, source, source_vocab,
                                    target_vocab, device)

        output_sentences.append(' '.join(output_words))

    return output_sentences


def translate_random_sentence(encoder, decoder, pairs, source_vocab,
                              target_vocab, device, n=1):
    """Translate n random sentence pairs and print their translation."""
    for i in range(n):
        source, target = random.choice(pairs)
        print(f'> {source}\n= {target}')

        output_words, _ = translate(encoder, decoder, source, source_vocab,
                                    target_vocab, device)
        translation = ' '.join(output_words)
        print(f'< {translation}\n')
