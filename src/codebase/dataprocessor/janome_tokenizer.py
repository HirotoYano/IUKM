import re

from janome.tokenizer import Tokenizer


def janome_tokenizer(clause):
    t = Tokenizer(wakati=True)

    replaced_clause = re.sub(r"[【】]", " ", clause)
    replaced_clause = re.sub(r"[（）()]", " ", replaced_clause)
    replaced_clause = re.sub(r"[［］\[\]]", " ", replaced_clause)
    replaced_clause = re.sub(r"[@＠]\w+", "", replaced_clause)
    replaced_clause = re.sub(r"\d+\.*\d*", "", replaced_clause)
    replaced_clause = re.sub("[、。]", "", replaced_clause)
    kana_re = re.compile("^[ぁ-ゖ]+$")

    token_list = list(t.tokenize(replaced_clause))
    token_list = [t for t in token_list if not kana_re.match(t)]

    return token_list
