from abc import ABC, abstractmethod
from language import add_alphabet_to_ocurrence_dict
from typing import List


class CharNotInVocabularyException(Exception):
    pass


class NgramModel(ABC):
    def __init__(self, vocab: int):
        self.n = 0
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

    def _vocab_safe_check(self, ngram: str):
        if self.vocab == 0 or self.vocab == 1:
            for char in ngram:
                # checking first level is sufficient, corpus already populated
                # with correct alphabet
                if char not in self.corpus:
                    raise CharNotInVocabularyException('Dismiss for vocab {}: "{}"'.format(self.vocab, ngram))
        elif self.vocab == 2:
            for char in ngram:
                if char not in self.corpus and char.isalpha():
                    self._spread_new_vocab_char(char)
                else:
                    raise CharNotInVocabularyException('Dismiss for vocab {}: "{}"'.format(self.vocab, ngram))

    def insert(self, ngrams: List[str]):
        for ngram in ngrams:
            try:
                self._vocab_safe_check(ngram)
                self._insert_ngram(ngram)
            except CharNotInVocabularyException as e:
                print(e)
                continue

    @abstractmethod
    def _build_corpus(self):
        pass

    @abstractmethod
    def _spread_new_vocab_char(self, char: str):
        pass

    @abstractmethod
    def _insert_ngram(self, ngram: str):
        pass


class UnigramModel(NgramModel):
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
        super(UnigramModel, self).__init__(vocab)
        self.n = 1
        pass

    def _build_corpus(self):
        self._build_one_level_vocab(self.corpus)

    def _spread_new_vocab_char(self, char: str):
        self.corpus[char] = 0

    def _insert_ngram(self, char: str):
        self.corpus[char] += 1


class BigramModel(NgramModel):
    """
    First-order Markov model

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
        super(BigramModel, self).__init__(vocab)
        self.n = 2
        pass

    def _build_corpus(self):
        self._build_one_level_vocab(self.corpus)
        for char in self.corpus:
            self.corpus[char] = {}
            self._build_one_level_vocab(self.corpus[char])

    def _spread_new_vocab_char(self, char: str):
        self.corpus[char] = {}
        self.corpus[char][char] = 0
        self._build_one_level_vocab(self.corpus[char])

    def _insert_ngram(self, bigram: str):
        self.corpus[bigram[0]][bigram[1]] += 1


class TrigramModel(NgramModel):
    """
    Second-order Markov model

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
        super(TrigramModel, self).__init__(vocab)
        self.n = 3
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

    def _spread_new_vocab_char(self, char: str):
        self.corpus[char] = {}
        self.corpus[char][char] = {}
        self.corpus[char][char][char] = 0
        self._build_one_level_vocab(self.corpus[char][char])

    def _insert_ngram(self, trigram: str):
        self.corpus[trigram[0]][trigram[1]][trigram[2]] += 1
