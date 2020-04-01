from abc import ABC, abstractmethod
from language import add_alphabet_to_ocurrence_dict


class Ngram(ABC):
    def __init__(self, vocab: int):
        self.corpus = {}
        self.vocab = vocab
        self._build_corpus()

    def _build_one_level_vocab(self, target_dict):
        if self.vocab == 0:
            add_alphabet_to_ocurrence_dict(False, target_dict)
        elif self.vocab == 1:
            add_alphabet_to_ocurrence_dict(False, target_dict)
            add_alphabet_to_ocurrence_dict(True, target_dict)
        elif self.vocab == 2:
            add_alphabet_to_ocurrence_dict(False, target_dict)
            add_alphabet_to_ocurrence_dict(True, target_dict)

    @abstractmethod
    def _build_corpus(self):
        pass

    @abstractmethod
    def insert(self, seq: str):
        # TODO: probably has the logic for isAlpha
        pass


class Unigram(Ngram):
    """
    Bag of words (or character in this case)

    corpus = {
        'a': 8,
        'b': 5,
        'c': 19,
        [...]
    }
    """

    def __init__(self, vocab: int):
        super(Unigram, self).__init__(vocab)
        pass

    def _build_corpus(self):
        self._build_one_level_vocab(self.corpus)

    def insert(self, seq: str):
        pass


class Bigram(Ngram):
    """
    corpus = {
        'a': {
            'a': 8,
            'b': 5,
            'c': 19,
            [...]
        },
        [...]
    }
    """

    def __init__(self, vocab: int):
        super(Bigram, self).__init__(vocab)
        pass

    def _build_corpus(self):
        self._build_one_level_vocab(self.corpus)
        for char in self.corpus:
            self.corpus[char] = {}
            self._build_one_level_vocab(self.corpus[char])

    def insert(self, seq: str):
        pass


class Trigram(Ngram):
    """
        corpus = {
            'a': {
                'a': {
                        'a': 8,
                        'b': 5,
                        'c': 19,
                        [...]
                    },
                [...]
            },
            [...]
        }
    """

    def __init__(self, vocab):
        super(Trigram, self).__init__(vocab)
        pass

    def _build_corpus(self):
        self._build_one_level_vocab(self.corpus)
        for char_f in self.corpus:
            self.corpus[char_f] = {}
            self._build_one_level_vocab(self.corpus[char_f])
            for char_s in self.corpus[char_f]:
                self.corpus[char_f][char_s] = {}
                self._build_one_level_vocab(self.corpus[char_f][char_s])
        pass

    def insert(self, seq: str):
        pass
