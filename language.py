import string

LANGUAGES = ['eu', 'ca', 'gl', 'es', 'en', 'pt']


def add_alphabet_to_ocurrence_dict(is_uppercase: bool, occ_dict):
    for letter in string.ascii_uppercase if is_uppercase else string.ascii_lowercase:
        occ_dict[letter] = 0
