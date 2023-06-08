from gensim.corpora.dictionary import Dictionary


def make_dict(token_lists: list):
    dictionary = Dictionary(token_lists)
    corpus = [dictionary.doc2bow(token_list) for token_list in token_lists]

    return dictionary, corpus
