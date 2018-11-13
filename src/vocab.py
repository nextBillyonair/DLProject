from itertools import takewhile

"""File for handling vocabulary building"""

# Tokens
START_OF_SENTENCE = '<SOS>'
START_OF_SENTENCE_INDEX = 0
END_OF_SENTENCE = '<EOS>'
END_OF_SENTENCE_INDEX = 1

class Bivocabulary:
    @classmethod
    def create(cls, source_language, target_language):
        source = Vocabulary(source_language)
        target = Vocabulary(target_language)

        return cls(source, target)

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def add_sentence_pair(self, pair):
        source, target = pair.split('|||', 1)

        source_indices = self.source.add_sentence(source)
        target_indices = \
            self.target.add_sentence(f'{target} {END_OF_SENTENCE}')

        return source_indices, target_indices


class Vocabulary:
    def __init__(self, language):
        self._language = language
        self._word_indices = {}
        self._index_words = {}
        self._next_index = 0

        self._add_word(START_OF_SENTENCE)
        self._add_word(END_OF_SENTENCE)

        assert self._word_indices[START_OF_SENTENCE] == START_OF_SENTENCE_INDEX
        assert self._word_indices[END_OF_SENTENCE] == END_OF_SENTENCE_INDEX

    def add_sentence(self, sentence):
        indices = tuple(self._add_word(word) for word in sentence.split())
        assert len(indices) > 0, 'empty sentence'
        return indices

    def _add_word(self, word):
        if word not in self._word_indices:
            self._word_indices[word] = self._next_index
            self._index_words[self._next_index] = word
            self._next_index += 1

        return self._word_indices[word]

    def sentence_from(self, tensor):
        assert len(tensor.size()) == 1

        trimmed_sentence = takewhile(lambda i: i != END_OF_SENTENCE_INDEX,
                                     (t.item() for t in tensor))
        sentence = ' '.join(self._index_words[i] for i in trimmed_sentence)

        return sentence

    def sentences_from(self, tensor):
        assert len(tensor.size()) == 2

        for sentence_tensor in tensor:
            yield self.sentence_from(sentence_tensor)

    def __len__(self):
        return len(self._word_indices)
