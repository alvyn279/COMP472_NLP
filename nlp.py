import argparse

parser = argparse.ArgumentParser(
    description='Naive Bayes Classifier for Tweet Language Detection',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('v',
                    help="""Vocabulary to use
                    0:[a-z],
                    1:[a-z, A-Z],
                    2:[a-z, A-Z] + all characters accepted by built-in isalpha()
                    """,
                    type=int)
parser.add_argument('n',
                    help="""Size of n-grams
                    1:character unigram (bag of words),
                    2:character bigrams,
                    3:character trigrams
                    """,
                    type=int)
parser.add_argument('delta',
                    help='Smoothing value δ used for additive smoothing',
                    type=float)
parser.add_argument('training_file',
                    help='Path to training file for the language models',
                    type=str)
parser.add_argument('testing_file',
                    help='Path to testing file for the language models',
                    type=str)


def main():
    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
