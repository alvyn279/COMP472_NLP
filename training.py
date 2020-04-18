from ngrams import NgramModel, CharNotInVocabularyException
from nltk.tokenize import word_tokenize
from typing import List
from language import LANGUAGES
import copy
import math

count = 0
# unicode = 17 planes of 2**16 symbols
for codepoint in range(17 * 2 ** 16):
    ch = chr(codepoint)
    if ch.isalpha():
        count = count + 1

IS_ALPHA_COUNT = count

BLACKLIST = ['http', 'https']
BLACKLIST_SET = set(BLACKLIST)


class NgramTrainingModel:
    """
    Operates on the models, such as frequency to probability calculator
    """

    def __init__(self, language: str, ngram_model: NgramModel):
        self.language = language
        self.ngram_model = ngram_model
        self.smoothing = 0
        self.docs_for_this_model = 0  # e.g.: documents marked 'es'
        self.class_size = 0  # number of ngrams inserted in corpus
        self.total_doc_count = 0  # total number of documents scanned
        self.probabilities = {}
        self.size_of_vocab = 0
        self.prior = 0.0
        self.score = 0.0
        self.non_existing_char_prob = 0.0

    def _ngrams_list(self, tweet: str):
        """
        Divides the tweet into all the possible contiguous sequences of characters of
         length [unigram:1, bigram:2, trigram:3]
        """
        return list(tweet[j:j + self.ngram_model.n] for j in range(0, len(tweet) - (self.ngram_model.n - 1)))

    def _compute_prob_value(self, occurence: int):
        """
        Returns log-ified (base 10) value to be added directly to the class score
        """
        if occurence + self.smoothing == 0:
            return 0
        return math.log10((occurence + self.smoothing) / (self.class_size + self.size_of_vocab))

    def insert(self, tweet: str):
        """
        Given a tweet, tries to insert all possible ngrams, while updating class size
        """
        self.docs_for_this_model += 1
        self.class_size += self.ngram_model.insert(self._ngrams_list(tweet))

    def post_parse(self, total_doc_count, smoothing):
        """
        Performed before compute()
        """
        self.smoothing = smoothing
        self.total_doc_count = total_doc_count
        self.size_of_vocab = len(self.ngram_model.corpus)
        if self.ngram_model.vocab == 2:
            self.size_of_vocab += IS_ALPHA_COUNT
            self.non_existing_char_prob = self._compute_prob_value(0)
        self.probabilities = copy.deepcopy(self.ngram_model.corpus)
        self.prior = math.log10(self.docs_for_this_model / self.total_doc_count)

    def compute(self):
        """
        Computes probabilities over the held copy of the corpus of the language model
        """
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
        Computes score for given tweet for the current class (language)
        """
        score = self.prior
        for ngram in self._ngrams_list(tweet):
            try:
                self.ngram_model.vocab_safe_check(ngram, False)
                score += self.ngram_model.get_prob(ngram, self.probabilities)
            except CharNotInVocabularyException as e:
                score += self.non_existing_char_prob

        return score


class TFIDFWithStopWordTrainingModel:
    """
    Simple bag of words model with increased weight given to stop words not that anymore
    """

    def __init__(self, language: str, stop_words: List[str]):
        self.language = language
        self.corpus = {}
        self.stop_words = stop_words
        self.word_occ_in_other_models = {}
        self.weights = {}

    def insert(self, single_word: str):
        """
        Adds occurence value to bag of words
        Features:
            - stop-word/non stop word
            - is an alphanumerical word (denies special characters)
            - omits single character words
            - word should not be in blacklist
        """
        non_stop_word_value = 1
        stop_word_value = 5

        lower_single_word = single_word.lower()
        if lower_single_word.isalnum() \
                and len(lower_single_word) != 1 \
                and lower_single_word not in BLACKLIST_SET:
            if lower_single_word not in self.stop_words:
                if lower_single_word in self.corpus:
                    self.corpus[lower_single_word] += non_stop_word_value
                else:
                    self.corpus[lower_single_word] = non_stop_word_value
            else:
                # Added value to stop words
                if lower_single_word in self.corpus:
                    self.corpus[lower_single_word] += stop_word_value
                else:
                    self.corpus[lower_single_word] = stop_word_value

    def set_word_occ_in_other_models(self, word_occ):
        """
        Sets the value for the dict of word occurence in other models for the current
        language. Max value of len(LANGUAGES)
        """
        self.word_occ_in_other_models = word_occ

    def compute(self):
        """
        Computes td-idf for ocurrence value of every word in the corpus
        """
        for word in self.corpus:
            self.weights[word] = (1 + math.log10(self.corpus[word])) * \
                                 math.log10(len(LANGUAGES) / self.word_occ_in_other_models[word])

    def test(self, tweet: str):
        """
        Computes score for given tweet for the tf-idf stop word training model
        """
        score = 0
        text_tokens = word_tokenize(tweet)
        for text_token in text_tokens:
            lower_text_token = text_token.lower()
            if lower_text_token in self.weights:
                score += self.weights[lower_text_token]
        return score


class Score:
    """
    Holds the score attributes obtained for a single tweet after running against model
    """
    def __init__(self, tweet_id, score, guessed_lang, actual_lang):
        self.tweet_id = tweet_id
        self.score = score
        self.guessed_lang = guessed_lang
        self.actual_lang = actual_lang
        self.is_correct = self.guessed_lang == self.actual_lang

    def __str__(self):
        return "\n{}\t{}\t{}\t{}\t{}".format(
            self.tweet_id,
            self.guessed_lang,
            self.score,
            self.actual_lang,
            "correct" if self.is_correct else "wrong"
        )


class ClassScore:
    """
    For a given class (i.e: language), holds stats regarding the parsed data
    """
    def __init__(self, class_count: int):
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.count = class_count
