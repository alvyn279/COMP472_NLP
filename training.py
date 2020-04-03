from ngrams import NgramModel
import copy
import math


class LanguageTrainingModel:
    """
    Operates on the models, such as frequency to probability calculator
    """

    def __init__(self, language: str, ngram_model: NgramModel):
        self.language = language
        self.ngram_model = ngram_model
        self.smoothing = 0
        self.class_size = 0
        self.total_doc_count = 0  # needed to calculate prior
        self.probabilities = {}
        self.size_of_vocab = 0
        self.prior = 0.0
        self.score = 0.0

    def _ngrams_list(self, tweet: str):
        return list(tweet[j:j + self.ngram_model.n] for j in range(0, len(tweet) - (self.ngram_model.n - 1)))

    def _compute_prob_value(self, occurence: int):
        """
        Returns log-ified (base 10) value to be added directly to the class score
        """
        return math.log10((occurence + self.smoothing) / (self.class_size + self.size_of_vocab))

    def insert(self, tweet: str):
        self.class_size += 1
        self.ngram_model.insert(self._ngrams_list(tweet))

    def post_parse(self, total_doc_count, smoothing):
        """
        Performed before compute()
        """
        self.smoothing = smoothing
        self.total_doc_count = total_doc_count
        self.size_of_vocab = len(self.ngram_model.corpus)
        self.probabilities = copy.deepcopy(self.ngram_model.corpus)
        self.prior = math.log10(self.class_size / self.total_doc_count)

    def compute(self):
        for char_1 in self.probabilities:
            if isinstance(self.probabilities[char_1], int):
                self.probabilities[char_1] = self._compute_prob_value(self.probabilities[char_1])
                continue

            for char_2 in self.probabilities[char_1]:
                if isinstance(self.probabilities[char_1][char_2], int):
                    self.probabilities[char_1][char_2] = \
                        self._compute_prob_value(self.probabilities[char_1][char_2])
                    continue

                for char_3 in self.probabilities[char_1][char_2]:
                    if isinstance(self.probabilities[char_1][char_2][char_3], int):
                        self.probabilities[char_1][char_2][char_3] = \
                            self._compute_prob_value(self.probabilities[char_1][char_2][char_3])

    def test(self, tweet: str):
        """
        Computes score for tested tweet for the current class
        """
        pass
