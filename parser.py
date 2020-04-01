from language import Language
from ngrams import Ngram


class DataParser:
    """
    Parses the data contained in the files given as input
    Builds models by populating the corpus
    """

    def __init__(self, input_file: str, ngram_size: int):
        self.input_file = input_file
        self.models = {}  # Dict[Language: LanguageTrainingModel]

        # TODO: instantiate concrete ngrams

    def parse(self):
        pass
