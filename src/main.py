from itertools import accumulate
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import random
import time
import torch
from torch.optim import Adam
from torch.nn import NLLLoss
import logging

from args import get_args
from conversion import tensors_from_pair
from vocab import make_vocabs, split_lines, clean
from encoders import build_encoder
from decoders import build_decoder
# from evaluate import translate_random_sentence, translate_sentences
from checkpoints import save_checkpoint, print_checkpoint, bleu_checkpoint

"""This is the Main File for our Seq2Seq Training"""

def main():

    args = get_args()

    # Load checkpoint?
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        starting_iteration = state['iteration'] + 1
    else:
        starting_iteration = 1

    # make vocab
    source_vocab = target_vocab = None
    if args.reverse:
        target_vocab, source_vocab = \
            make_vocabs(args.target_lang, args.source_lang)
    else:
        source_vocab, target_vocab = \
            make_vocabs(args.source_lang, args.target_lang)

    # Get Encoder
    encoder = build_encoder(source_vocab.n_words, args.hidden_size,
                            mode=args.encoder_mode, dropout=args.lstm_dropout,
                            num_layers=args.encoder_layers, device=args.device
                            ).to(args.device)

    # Get Decoder
    decoder = build_decoder(args.hidden_size, target_vocab.n_words,
                            mode=args.decoder_mode,
                            attention=args.attention_mode,
                            embedding_dropout=args.embedding_dropout,
                            lstm_dropout=args.lstm_dropout,
                            num_layers=args.decoder_layers,
                            device=args.device).to(args.device)

    # Encoder/decoder weights are randomly initialized, so load these if
    # checkpointed
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['encoder_state'])
        decoder.load_state_dict(state['decoder_state'])

    # Read in data files
    train_pairs = split_lines(args.train_file, args.reverse)
    dev_pairs = split_lines(args.dev_file, args.reverse)
    test_pairs = split_lines(args.test_file, args.reverse)

    # Set up optimizer/loss
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = Adam(params, lr=args.initial_learning_rate)
    criterion = NLLLoss()

    # Load optimizer state from checkpoint
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['optimizer_state'])


    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    cum_weights = list(range(1, len(train_pairs) + 1))

    for iteration in range(starting_iteration, args.iterations + 1):
        # get a random pair and convert
        pair = random.choices(train_pairs, cum_weights=cum_weights)[0]
        input_tensor, target_tensor = \
            tensors_from_pair(source_vocab, target_vocab, pair, args.device)

        # train
        loss = train(input_tensor, target_tensor, encoder, decoder, optimizer,
                     criterion, args.device)
        print_loss_total += loss

        if iteration % args.checkpoint_every == 0:
            save_checkpoint(iteration, encoder, decoder, optimizer)

        if iteration % args.print_every == 0:
            print_checkpoint(args, print_loss_total, start, iteration,
                             encoder, decoder, dev_pairs,
                             source_vocab, target_vocab)

    # END OF TRAIN

    # TEST

    # Translate entire test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs,
                                               source_vocab, target_vocab,
                                               args.device)

    # Write to out
    with open(args.out_file, 'w') as f:
        for sentence in translated_sentences:
            print(clean(sentence), file=f)


if __name__ == '__main__':
    main()
