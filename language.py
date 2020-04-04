import string

LANGUAGES = ['eu', 'ca', 'gl', 'es', 'en', 'pt']

LANGUAGE_DICT = {
    'eu': 'basque',
    'ca': 'catalan',
    'gl': 'galacian',
    'es': 'spanish',
    'en': 'english',
    'pt': 'portuguese'
}


def add_alphabet_to_ocurrence_dict(is_uppercase: bool, occ_dict):
    for letter in string.ascii_uppercase if is_uppercase else string.ascii_lowercase:
        occ_dict[letter] = 0
