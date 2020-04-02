from language import LANGUAGES
from training import LanguageTrainingModel
from ngrams import UnigramModel, BigramModel, TrigramModel

from typing import List


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

    def parse(self):
        try:
            f = open(self.input_file, "r")
        except FileNotFoundError as e:
            print(e)
            print("Please input a file that exists.")
            exit(1)

        lines: List[str] = f.readlines()
        for line in lines:
            line_info: List[str] = line.split('\t')
            parsed_user_id = line_info[0]
            parsed_username = line_info[1]
            parsed_language = line_info[2]
            parsed_tweet_content = line_info[3]

            self.models[parsed_language].insert(parsed_tweet_content)

