from typing import List

from tqdm import tqdm

from codebase.computation.calc_tf_idf import calc_tf_idf


def create_doc_representation(docs: List[List[str]], word_list: List[str]) -> List[List[float]]:
    tf_idf = calc_tf_idf(docs, word_list)

    d_real: List[List[float]] = []

    print("=== 文章表現の抽出 ===")
    for i, _ in enumerate(tqdm(tf_idf, leave=False)):
        d: List[float] = []
        for j, _ in enumerate(tqdm(word_list, leave=False)):
            d.append(tf_idf[i][j] / sum(tf_idf[i]))
        d_real.append(d)

    return d_real
