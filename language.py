import enum
import string


class Language(enum.Enum):
    EU = 'eu'
    CA = 'ca'
    GL = 'gl'
    ES = 'es'
    EN = 'en'
    PT = 'pt'


def add_alphabet_to_ocurrence_dict(is_uppercase: bool, corpus):
    for letter in string.ascii_uppercase if is_uppercase else string.ascii_uppercase:
        corpus[letter] = 0
