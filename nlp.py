import argparse
from ngram_parser import NgramTrainingDataParser, NgramTestParser
from byom_parser import TFIDFWithStopWordTrainingParser, StopWordTestParser

parser = argparse.ArgumentParser(
    description='Naive Bayes Classifier for Tweet Language Detection',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('v',
                    help="""Vocabulary to use
                    0:[a-z],
                    1:[a-z, A-Z],
                    2:[a-z, A-Z] + all characters accepted by built-in isalpha(),
                    -1: BYOM
                    """,
                    type=int)
parser.add_argument('n',
                    help="""Size of n-grams
                    1:character unigram (bag of words),
                    2:character bigrams,
                    3:character trigrams,
                    -1: BYOM
                    """,
                    type=int)
parser.add_argument('delta',
                    help='Smoothing value δ used for additive smoothing, -1: BYOM',
                    type=float)
parser.add_argument('training_file',
                    help='Path to training file for the language models',
                    type=str)
parser.add_argument('testing_file',
                    help='Path to testing file for the language models',
                    type=str)


def main():
    args = parser.parse_args()

    if args.v == -1 and args.n == -1 and args.delta == -1:
        tf_idf_stop_word_model_training_parser: TFIDFWithStopWordTrainingParser = TFIDFWithStopWordTrainingParser(
            args.training_file
        )
        tf_idf_stop_word_model_training_parser.parse()

        stop_word_test_parser: StopWordTestParser = StopWordTestParser(
            tf_idf_stop_word_model_training_parser,
            args.testing_file
        )
        stop_word_test_parser.parse()
    else:
        training_data_parser: NgramTrainingDataParser = NgramTrainingDataParser(
            args.training_file,
            args.n,
            args.v,
            args.delta
        )
        training_data_parser.parse()

        test_parser: NgramTestParser = NgramTestParser(
            training_data_parser,
            args.testing_file
        )
        test_parser.parse()


if __name__ == '__main__':
    main()
