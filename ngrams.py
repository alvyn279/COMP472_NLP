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
        self.extra_vocab_chars = set()
        self._build_corpus()

    def _build_one_level_vocab(self, target_dict):
        """
        Builds at current level of dic
        """
        if self.vocab == 0:
            add_alphabet_to_ocurrence_dict(False, target_dict)
        else:
            add_alphabet_to_ocurrence_dict(False, target_dict)
            add_alphabet_to_ocurrence_dict(True, target_dict)

        if self.vocab == 2:
            for char in self.extra_vocab_chars:
                target_dict[char] = 0

    def vocab_safe_check(self, ngram: str, update=True):
        """
        Checks whether or not an n-gram can be inserted into the corpus
        Updating the corpus with a new char can be optionally ignores
        Raises CharNotInVocabularyException if not possible.
        """
        if self.vocab == 0 or self.vocab == 1:
            for char in ngram:
                # checking first level is sufficient, corpus already populated
                # with correct alphabet
                if char not in self.corpus:
                    raise CharNotInVocabularyException('Dismiss for vocab {}: "{}"'.format(self.vocab, ngram))
        elif self.vocab == 2:
            for char in ngram:
                if char.isalpha():
                    if char not in self.corpus and update:
                        self.extra_vocab_chars.add(char)
                        self._spread_new_vocab_char(char)
                    elif char not in self.corpus:
                        raise CharNotInVocabularyException('Dismiss for vocab {}: "{}"'.format(self.vocab, ngram))
                else:
                    raise CharNotInVocabularyException('Dismiss for vocab {}: "{}"'.format(self.vocab, ngram))

    def insert(self, ngrams: List[str]):
        """
        Inserts if possible an ngram in the corpus, returns the amount of ngrams inserted
        """
        count = 0
        for ngram in ngrams:
            try:
                self.vocab_safe_check(ngram)
                self._insert_ngram(ngram)
                count += 1
            except CharNotInVocabularyException as e:
                print(e)
                continue
        return count

    @abstractmethod
    def _build_corpus(self):
        """
        Builds initial corpus depending on type of n-gram
        """
        pass

    @abstractmethod
    def _spread_new_vocab_char(self, char: str):
        """
        Given a char, spreads its value as new keys in the corpus dict
        """
        pass

    @abstractmethod
    def _insert_ngram(self, ngram: str):
        """
        Inserts an n-gram by adding 1 to its ocurrence in the corpus
        """
        pass

    @staticmethod
    @abstractmethod
    def get_prob(ngram: str, probabilities):
        """
        Defines how to access a corpusfor an n-gram
        """
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

    @staticmethod
    def get_prob(ngram: str, probabilities):
        return probabilities[ngram[0]]


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
        self._build_one_level_vocab(self.corpus[char])

        for char_1 in self.corpus:
            self.corpus[char_1][char] = 0

    def _insert_ngram(self, bigram: str):
        self.corpus[bigram[0]][bigram[1]] += 1

    @staticmethod
    def get_prob(ngram: str, probabilities):
        return probabilities[ngram[0]][ngram[1]]


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

    def _build_corpus(self):
        self._build_one_level_vocab(self.corpus)
        for char_f in self.corpus:
            self.corpus[char_f] = {}
            self._build_one_level_vocab(self.corpus[char_f])
            for char_s in self.corpus[char_f]:
                self.corpus[char_f][char_s] = {}
                self._build_one_level_vocab(self.corpus[char_f][char_s])

    def _spread_new_vocab_char(self, char: str):
        self.corpus[char] = {}
        self._build_one_level_vocab(self.corpus[char])
        for char_2 in self.corpus[char]:
            self.corpus[char][char_2] = {}
            self._build_one_level_vocab(self.corpus[char][char_2])

        for char_1 in self.corpus:
            self.corpus[char_1][char] = {}
            self._build_one_level_vocab(self.corpus[char_1][char])

            for char_2 in self.corpus[char_1]:
                self.corpus[char_1][char_2][char] = 0

    def _insert_ngram(self, trigram: str):
        self.corpus[trigram[0]][trigram[1]][trigram[2]] += 1

    @staticmethod
    def get_prob(ngram: str, probabilities):
        return probabilities[ngram[0]][ngram[1]][ngram[2]]
