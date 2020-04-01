from ngrams import Ngram


class LanguageTrainingModel:
    def __init__(self, language: str, ngram: Ngram):
        self.language = language
        self.ngram = ngram
