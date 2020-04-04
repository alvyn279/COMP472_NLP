from language import LANGUAGES, LANGUAGE_DICT
from training import StopWordTrainingModel, Score, ClassScore

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nlp_tools.bkp_stop_words import BKP_STOP_WORDS

from typing import List

import os

REL_PATH_TO_TRACE = "./output/trace_{}_{}_{}.txt"
REL_PATH_TO_EVAL = "./output/eval_{}_{}_{}.txt"

language_stopwords = {}

for lang in LANGUAGES:
    try:
        language_stopwords[lang] = stopwords.words(LANGUAGE_DICT[lang])
    except IOError as e:
        language_stopwords[lang] = BKP_STOP_WORDS[lang]


class StopWordTrainingParser:
    """
    Parses the data contained in the files given as input
    Builds models by populating stop word corpus
    """

    def __init__(self, input_file: str):
        self.input_file: str = input_file
        self.models = {}  # Dict[Language: StopWordTrainingModel]
        self.language_stop_words = language_stopwords
        for language in LANGUAGES:
            self.models[language] = StopWordTrainingModel(language, self.language_stop_words[language])

    def parse(self):
        document_count = 0
        try:
            f = open(self.input_file, "r")
        except FileNotFoundError as e:
            print(e)
            print("Please input a file that exists.")
            exit(1)

        lines: List[str] = f.readlines()
        for line in lines:
            document_count += 1
            line_info: List[str] = line.split('\t')
            parsed_tweet_id = line_info[0]
            parsed_username = line_info[1]
            parsed_language = line_info[2]
            parsed_tweet_content = line_info[3]
            text_tokens = word_tokenize(parsed_tweet_content)
            for text_token in text_tokens:
                self.models[parsed_language].insert(text_token)


class StopWordTestParser:
    """
    Class that run test file against trained stop word model
    """

    def __init__(self, stop_word_training_parser: StopWordTrainingParser, input_test_file: str):
        self.stop_word_training_parser = stop_word_training_parser
        self.input_test_file = input_test_file
        self.count = 0
        self.results: List[List[Score]] = []
        self.trace_output: str = ''
        self.final_accuracy = 0.0
        self.final_macro_f1 = 0.0
        self.final_weighed_avg_f1 = 0.0
        self.class_scores = {}
        self.class_occ = {}

    def parse(self):
        try:
            f = open(self.input_test_file, "r")
        except FileNotFoundError as e:
            print(e)
            print("Please input a test file that exists.")
            exit(1)

        lines: List[str] = f.readlines()
        for line in lines:
            try:
                line_info: List[str] = line.split('\t')
                parsed_tweet_id = line_info[0]
                parsed_username = line_info[1]
                parsed_language = line_info[2]
                parsed_tweet_content = line_info[3]
            except IndexError as e:
                print('Skipped testing for: {}'.format(line))
                continue

            if parsed_language not in self.class_occ:
                self.class_occ[parsed_language] = 1
            else:
                self.class_occ[parsed_language] += 1

            self.count += 1
            scores_for_tweet = []
            for language in LANGUAGES:
                lang_score: float = self.stop_word_training_parser.models[language].test(parsed_tweet_content)
                scores_for_tweet.append(Score(
                    parsed_tweet_id,
                    lang_score,
                    language,
                    parsed_language
                ))

            sorted_scores = sorted(scores_for_tweet, key=lambda x: x.score, reverse=True)
            self.results.append(sorted_scores)

        self.process_results()

    def process_results(self):
        correct = 0
        result: List[Score]  # sorted in reverse order
        for result in self.results:
            self.trace_output += str(result[0])  # gets largest score
            correct += 1 if result[0].is_correct else 0

        self.final_accuracy = correct / len(self.results)
        self.trace_output += '\n\nAccuracy: {}'.format(self.final_accuracy)
        print(self.trace_output)
