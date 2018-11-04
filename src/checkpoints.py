import torch
import logging
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from evaluate import translate_random_sentence
from vocab import clean

"""File handles checkpoints for main"""

def save_checkpoint(iteration, encoder, decoder, optimizer):
    """Saves current state on iteration"""
    state = {'iteration': iteration,
             'encoder_state': encoder.state_dict(),
             'decoder_state': decoder.state_dict(),
             'optimizer_state': optimizer.state_dict()}

    path = "state_{:010d}".format(iteration)
    torch.save(state, path)
    logging.debug('wrote checkpoint to %s', path)


def print_checkpoint(args, print_loss_total, start, iteration,
                     encoder, decoder, dev_pairs,
                     source_vocab, target_vocab):
    """Prints a checkpoint, showing loss and 2 random translations"""

    print_loss_avg = print_loss_total / args.print_every
    print_loss_total = 0

    logging.info('time since start:%s (iter: %d - %.2f%%) '
                 'loss_avg:%.4f', time.time() - start, iteration,
                 float(iteration) / args.iterations * 100,
                 print_loss_avg)

    # Translate from the dev set
    translate_random_sentence(encoder, decoder, dev_pairs,
                              source_vocab, target_vocab, args.device,
                              n=2)

    logging.info('Translating dev sentences...')
    translated_sentences = \
        translate_sentences(encoder, decoder, dev_pairs, source_vocab,
                            target_vocab, args.device)

    references = [[clean(pair[1]).split(),]
                  for pair in dev_pairs[:len(translated_sentences)]]
    candidates = [clean(sent).split() for sent in translated_sentences]
    logging.info('Computing dev BLEU...')
    dev_bleu = corpus_bleu(references, candidates)
    logging.info('Dev BLEU score: %.2f', dev_bleu)


def bleu_checkpoint():
    """Saves current state on iteration"""
    pass
