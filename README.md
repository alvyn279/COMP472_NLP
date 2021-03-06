# COMP472_NLP
###### by Duy-Khoi Alvyn Le

### Naive Bayes classifier for Tweet Language Identification

Any language from Basque (eu), Catalan (ca), Galician (gl), Spanish (es), English (en), and Portuguese (pt).

Models implemented:
- **n-gram** (unigram, bigram, and trigram): ~ 60-70% accuracy
- **tf-idf with stop words**: ~ 90+% accuracy


### Running the program
Make sure to create an `output/` directory at root dir level.
```sh
# create virtual env
python3 -m venv venv
# activate env
source ./venv/bin/activate
# install requirements
pip install -r requirements.txt
# for usage
python nlp.py --help
# usage: nlp.py [-h] v n delta training_file testing_file
```

### References
I made use of a set of static stopwords other than from _nltk_'s sources for Basque, Galician, and Catalan languages. 
[This is the link](https://github.com/Xangis/extra-stopwords) to the GitHub repository.

