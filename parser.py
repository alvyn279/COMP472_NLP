from language import LANGUAGES
from training import LanguageTrainingModel
from ngrams import UnigramModel, BigramModel, TrigramModel


class DataParser:
    """
    Parses the data contained in the files given as input
    Builds models by populating the corpus
    """

    def __init__(self, input_file: str, ngram_size: int, vocabulary: int, smoothing: float):
        self.input_file: str = input_file
        self.ngram_size: int = ngram_size
        self.vocabulary: int = vocabulary
        self.smoothing: float = smoothing
        self.models = {}  # Dict[Language: LanguageTrainingModel]

        for lang in LANGUAGES:
            if ngram_size == 1:
                self.models[lang] = LanguageTrainingModel(lang, UnigramModel(vocabulary))
            elif ngram_size == 2:
                self.models[lang] = LanguageTrainingModel(lang, BigramModel(vocabulary))
            elif ngram_size == 3:
                self.models[lang] = LanguageTrainingModel(lang, TrigramModel(vocabulary))

        print('lol')

    def parse(self):
        pass
