from language import LANGUAGES, LANGUAGE_DICT
from training import TFIDFWithStopWordTrainingModel, Score, ClassScore

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nlp_tools.bkp_stop_words import BKP_STOP_WORDS

from typing import List

import os

REL_PATH_TO_TRACE = "./output/trace_my_model.txt"
REL_PATH_TO_EVAL = "./output/eval_my_model.txt"

language_stopwords = {}

for lang in LANGUAGES:
    try:
        language_stopwords[lang] = set(stopwords.words(LANGUAGE_DICT[lang]))
    except IOError as e:
        language_stopwords[lang] = BKP_STOP_WORDS[lang]


class TFIDFWithStopWordTrainingParser:
    """
    Parses the data contained in the files given as input
    Builds models by populating stop word corpus
    """

    def __init__(self, input_file: str):
        self.input_file: str = input_file
        self.models = {}  # Dict[Language: TFIDFWithStopWordTrainingModel]
        self.language_stop_words = language_stopwords
        for language in LANGUAGES:
            self.models[language] = TFIDFWithStopWordTrainingModel(language, self.language_stop_words[language])

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

        self.byom_post_parse()

    def byom_post_parse(self):
        """
        Populates IDF with number of occurences of words in other languages training corpus
        and triggers tf-idf weight calculation
        """
        for language in self.models:
            word_occ_in_other_models = {}

            for other_language in self.models:
                if other_language == language:
                    continue

                for word in self.models[language].corpus:
                    if word not in word_occ_in_other_models:
                        word_occ_in_other_models[word] = 1  # at least 1 to avoid division by zero

                    if word in self.models[other_language].corpus:
                        word_occ_in_other_models[word] += 1

            self.models[language].set_word_occ_in_other_models(word_occ_in_other_models)

        for language in self.models:
            self.models[language].compute()


class StopWordTestParser:
    """
    Class that run test file against trained stop word model
    """

    def __init__(self, stop_word_training_parser: TFIDFWithStopWordTrainingParser, input_test_file: str):
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

    def _output_to_trace_file(self):
        cur_dir = os.path.dirname(__file__)
        abs_trace_path = os.path.join(cur_dir, REL_PATH_TO_TRACE)
        trace_f = open(abs_trace_path, "w+")
        trace_f.write(self.trace_output)
        trace_f.close()

    def _output_to_eval_file(self):
        cur_dir = os.path.dirname(__file__)
        abs_eval_path = os.path.join(cur_dir, REL_PATH_TO_EVAL)

        eval_f = open(abs_eval_path, "w+")
        eval_out = '\n'
        eval_out += '{}\n'.format(str(self.final_accuracy))
        eval_precision = ''
        eval_recall = ''
        eval_f1 = ''
        for lang_ in LANGUAGES:
            if lang_ in self.class_scores:
                eval_precision += '{}\t'.format(self.class_scores[lang_].precision)
                eval_recall += '{}\t'.format(self.class_scores[lang_].recall)
                eval_f1 += '{}\t'.format(self.class_scores[lang_].f1)

        eval_out += '{}\n{}\n{}\n'.format(eval_precision, eval_recall, eval_f1)
        eval_out += '{}\t{}\n'.format(self.final_macro_f1, self.final_weighed_avg_f1)
        eval_f.write(eval_out)
        eval_f.close()

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
        self._output_to_trace_file()
        self.run_stats()

    def run_stats(self):
        """
        Accuracy, per-class precision, per-class recall, per-class F1 measure,
        macro-F1, weighed-average-F1
        """
        class_scores_ = {}
        for lang_ in LANGUAGES:
            if lang_ in self.class_occ:
                class_scores_[lang_] = ClassScore(self.class_occ[lang_])

        result_list: List[Score]  # sorted in reverse order
        for result_list in self.results:
            for count, score in enumerate(result_list):
                first = count == 0
                correct = score.is_correct

                if score.guessed_lang in class_scores_:
                    if first and correct:
                        class_scores_[score.guessed_lang].true_positive += 1
                    elif first and not correct:
                        class_scores_[score.guessed_lang].false_positive += 1
                    elif not first and correct:
                        class_scores_[score.guessed_lang].false_negative += 1
                    elif not first and not correct:
                        class_scores_[score.guessed_lang].true_negative += 1

        for lang_ in class_scores_:
            if class_scores_[lang_].true_positive != 0 or class_scores_[lang_].false_positive != 0:
                class_scores_[lang_].precision = class_scores_[lang_].true_positive / \
                                                 (class_scores_[lang_].true_positive + class_scores_[
                                                     lang_].false_positive)

            if class_scores_[lang_].true_positive != 0 or class_scores_[lang_].false_negative != 0:
                class_scores_[lang_].recall = class_scores_[lang_].true_positive \
                                              / (class_scores_[lang_].true_positive + class_scores_[
                    lang_].false_negative)

            if class_scores_[lang_].precision != 0 or class_scores_[lang_].recall != 0:
                class_scores_[lang_].f1 = 2 * ((class_scores_[lang_].precision * class_scores_[lang_].recall)
                                               / (class_scores_[lang_].precision + class_scores_[lang_].recall))

        self.class_scores = class_scores_
        self.final_macro_f1 = sum([class_scores_[lang_].f1 for lang_ in class_scores_]) / len(class_scores_)
        self.final_weighed_avg_f1 = sum([class_scores_[lang].f1 * class_scores_[lang_].count
                                         for lang_ in class_scores_]) / self.count

        self._output_to_eval_file()
