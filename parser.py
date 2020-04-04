from language import LANGUAGES
from training import LanguageTrainingModel, Score, ClassScore
from ngrams import UnigramModel, BigramModel, TrigramModel

from typing import List

import os

REL_PATH_TO_TRACE = "./output/trace_{}_{}_{}.txt"
REL_PATH_TO_EVAL = "./output/eval_{}_{}_{}.txt"


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
            self.models[parsed_language].insert(parsed_tweet_content)

        for model_lang in self.models:
            self.models[model_lang].post_parse(document_count, self.smoothing)

        self.naive_bayes()

    def naive_bayes(self):
        for model_lang in self.models:  # for each class
            self.models[model_lang].compute()


class TestParser:
    """
    Class that runs trained models against a test file
    Precondition: DataParser object created and trained (parse() function was ran)
    """

    def __init__(self, parser: DataParser, input_test_file: str):
        self.training_parser = parser
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
        abs_trace_path = os.path.join(cur_dir, REL_PATH_TO_TRACE.format(self.training_parser.vocabulary,
                                                                        self.training_parser.ngram_size,
                                                                        self.training_parser.smoothing))
        trace_f = open(abs_trace_path, "w+")
        trace_f.write(self.trace_output)
        trace_f.close()

    def _output_to_eval_file(self):
        cur_dir = os.path.dirname(__file__)
        abs_eval_path = os.path.join(cur_dir, REL_PATH_TO_EVAL.format(self.training_parser.vocabulary,
                                                                      self.training_parser.ngram_size,
                                                                      self.training_parser.smoothing))

        eval_f = open(abs_eval_path, "w+")
        eval_out = '\n'
        eval_out += '{}\n'.format(str(self.final_accuracy))
        eval_precision = ''
        eval_recall = ''
        eval_f1 = ''
        for lang in LANGUAGES:
            eval_precision += '{}\t'.format(self.class_scores[lang].precision)
            eval_recall += '{}\t'.format(self.class_scores[lang].recall)
            eval_f1 += '{}\t'.format(self.class_scores[lang].f1)

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
            for lang in LANGUAGES:
                lang_score: float = self.training_parser.models[lang].test(parsed_tweet_content)
                scores_for_tweet.append(Score(
                    parsed_tweet_id,
                    lang_score,
                    lang,
                    parsed_language
                ))

            sorted_scores = sorted(scores_for_tweet, key=lambda x: x.score, reverse=True)
            self.results.append(sorted_scores)

        self.process_results()
        self.run_stats()
        self._output_to_eval_file()

    def process_results(self):
        correct = 0
        result: List[Score]  # sorted in reverse order
        for result in self.results:
            self.trace_output += str(result[0])  # gets largest score

            correct += 1 if result[0].is_correct else 0
        self.final_accuracy = correct / len(self.results)
        self.trace_output += '\n\nAccuracy: {}'.format(self.final_accuracy)
        self._output_to_trace_file()

    def run_stats(self):
        """
        Accuracy, per-class precision, per-class recall, per-class F1 measure,
        macro-F1, weighed-average-F1
        """
        class_scores_ = {}
        for lang in LANGUAGES:
            class_scores_[lang] = ClassScore(self.class_occ[lang])

        result_list: List[Score]  # sorted in reverse order
        for result_list in self.results:
            for count, score in enumerate(result_list):
                first = count == 0
                correct = score.is_correct

                if first and correct:
                    class_scores_[score.guessed_lang].true_positive += 1
                elif first and not correct:
                    class_scores_[score.guessed_lang].false_positive += 1
                elif not first and correct:
                    class_scores_[score.guessed_lang].false_negative += 1
                elif not first and not correct:
                    class_scores_[score.guessed_lang].true_negative += 1

        for lang in class_scores_:
            if class_scores_[lang].true_positive != 0 or class_scores_[lang].false_positive != 0:
                class_scores_[lang].precision = class_scores_[lang].true_positive / \
                                                (class_scores_[lang].true_positive + class_scores_[lang].false_positive)

            if class_scores_[lang].true_positive != 0 or class_scores_[lang].false_negative != 0:
                class_scores_[lang].recall = class_scores_[lang].true_positive \
                                             / (class_scores_[lang].true_positive + class_scores_[lang].false_negative)

            if class_scores_[lang].precision != 0 or class_scores_[lang].recall != 0:
                class_scores_[lang].f1 = 2 * ((class_scores_[lang].precision * class_scores_[lang].recall)
                                              / (class_scores_[lang].precision + class_scores_[lang].recall))

        self.class_scores = class_scores_
        self.final_macro_f1 = sum([class_scores_[lang].f1 for lang in class_scores_]) / len(class_scores_)
        self.final_weighed_avg_f1 = sum([class_scores_[lang].f1 * class_scores_[lang].count
                                         for lang in class_scores_]) / self.count
