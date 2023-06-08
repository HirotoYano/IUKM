import re

from sudachipy import Dictionary


def sudachi_tokenizer(clause):
    dictionary = Dictionary()
    tokenizer = dictionary.create()

    replaced_clause = re.sub(r"[【】]", " ", clause)
    replaced_clause = re.sub(r"[（）()]", " ", replaced_clause)
    replaced_clause = re.sub(r"[［］\[\]]", " ", replaced_clause)
    replaced_clause = re.sub(r"[@＠]\w+", "", replaced_clause)
    replaced_clause = re.sub(r"\d+\.*\d*", "", replaced_clause)
    replaced_clause = re.sub("[、。]", "", replaced_clause)
    kana_re = re.compile("^[ぁ-ゖ]+$")
    beeline_re = re.compile("^.$")

    morphemes = [
        (
            m.surface(),
            m.dictionary_form(),
            m.reading_form(),
            m.part_of_speech(),
        )
        for m in tokenizer.tokenize(replaced_clause)
    ]
    # morphemes = tokenizer.tokenize(replaced_clause)
    _len = len(morphemes)
    token_list = []

    for i in range(_len):
        if morphemes[i][3][0] == "名詞":
            # token_list = [m[0] for m in morphemes]
            token_list.append(morphemes[i][0])
    # token_list = [m.surface() for m in morphemes]
    token_list = [t for t in token_list if not kana_re.match(t)]
    token_list = [t for t in token_list if not beeline_re.match(t)]

    return token_list
