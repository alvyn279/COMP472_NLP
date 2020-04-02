from ngrams import NgramModel


class LanguageTrainingModel:
    def __init__(self, language: str, ngram_model: NgramModel):
        self.language = language
        self.ngram_model = ngram_model

    def _ngrams_list(self, tweet: str):
        return list(tweet[j:j + self.ngram_model.n] for j in range(0, len(tweet) - (self.ngram_model.n - 1)))

    def insert(self, tweet: str):
        self.ngram_model.insert(self._ngrams_list(tweet))
