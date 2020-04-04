# Content extracted from https://github.com/Xangis/extra-stopwords

BKP_STOP_WORDS = {
    'eu': [],
    'gl': [],
    'ca': []
}

for bkp_lang in BKP_STOP_WORDS:
    try:
        f = open('nlp_tools/{}'.format(bkp_lang), "r")
    except FileNotFoundError as e:
        print(e)
        print("Please add corresponding backup stop word file")
        exit(1)

    lines = f.readlines()
    bkp_lang_stop_words = []
    for line in lines:
        bkp_lang_stop_words.append(line.strip().split('\t')[0])
    BKP_STOP_WORDS[bkp_lang] = bkp_lang_stop_words
