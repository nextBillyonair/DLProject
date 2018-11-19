import torch
import random
from torch.nn import Dropout, Embedding, Linear, LSTM, Module
from torch.nn.functional import relu, log_softmax

from vocab import START_OF_SENTENCE_INDEX, END_OF_SENTENCE_INDEX
from encoders import build_encoder
from decoders import build_decoder
from utils import get_default_device


class Model:
    @classmethod
    def create_from_args(cls, args, vocab, max_length):
        device = get_default_device()

        encoder = build_encoder(args, vocab).to(device)

        decoder = build_decoder(args, vocab).to(device)

        return cls(encoder, decoder, max_length)


    def __init__(self, encoder, decoder, max_length):
        self._encoder = encoder
        self._decoder = decoder
        self._max_length = max_length


    def parameters(self):
        return list(self._encoder.parameters()) \
            + list(self._decoder.parameters())


    def train(self, source, target, target_lens, optimizer, criterion,
              teacher_force_percent=0.0):
        # tests
        assert len(source.size()) == 2
        assert len(target.size()) == 2
        assert len(target_lens.size()) == 1
        assert source.size(0) == target.size(0) == target_lens.size(0)

        device = get_default_device()
        optimizer.zero_grad()

        # Toggle training mode so dropout is enabled
        self._encoder.train()
        self._decoder.train()

        # Run input_tensor word-by-word through encoder
        encoder_output, hidden = self._encoder(source, None)

        # Run encoder output through decoder to build up target_tensor
        last_translated_tokens = torch.zeros(len(source), 1,
                                             dtype=torch.long, device=device)
        last_translated_tokens[:, 0] = START_OF_SENTENCE_INDEX

        loss = 0.0

        for i in range(target.size(1)):
            decoder_output, hidden = self._decoder(last_translated_tokens,
                                                   hidden, encoder_output)

            # Next input fed through is the most likely token from this
            # iteration
            _, most_likely_token_indices = decoder_output.topk(1)
            last_translated_tokens = \
                most_likely_token_indices.squeeze(1).detach()

            # Apply teacher forcing probabilistically per token (across batch)
            if random.random() < teacher_force_percent:
                last_translated_tokens = target[:, i:i+1]

            loss += criterion(decoder_output.squeeze(1), target[:, i],
                              i < target_lens)

        # Backprop
        loss.backward()
        optimizer.step()

        return loss.item()


    def translate(self, source):
        assert len(source.size()) == 2

        device = get_default_device()

        # Toggle eval mode to disable dropout
        self._encoder.eval()
        self._decoder.eval()

        with torch.no_grad():
            encoder_output, hidden = self._encoder(source, None)

            translation = torch.zeros(len(source), self._max_length + 1, 1,
                                      dtype=torch.long, device=device)
            translation[:, 0, 0] = START_OF_SENTENCE_INDEX

            for i in range(self._max_length):
                decoder_output, hidden = self._decoder(translation[:, i],
                                                       hidden, encoder_output)

                _, token_indices = decoder_output.data.topk(1)
                translation[:, i+1, 0] = token_indices.squeeze()

            # Slice off <SOS> token
            return translation[:, 1:, 0]







# EOF
