import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from codebase.computation.create_doc_representation import create_doc_representation
from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv
from codebase.dataprocessor.sudachi_tokenizer import sudachi_tokenizer

load_dotenv("/workspace/src/.env")
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    """
    ToDo: Encoderを作成
        ToDo: TF-IDFによる文書の重み付け
        ToDo: 各文書をV (語彙数) 次元の多項分布として表現
            ToDo: i番目の次元はi番目の単語と文書
        ToDo: 文書表現をEncoderに入力
            ToDo: S次元の意味空間に射影
            ToDo: K (トピック数) 次元のトピック空間に変換
    ToDo: Generatorを作成
        ToDo: ディリクレ事前分布 -> θf
        ToDo: θf -> Generator
            ToDo: θfをS次元空間に射影
            ToDo: oを単語分布dfに射影
    ToDo: Discriminatorを作成
        ToDo: 3層で構成
            ToDo: V + K次元の合同分布層
            ToDo: S次元表現層
            ToDo: 出力層
        ToDo: 入力：分布ペアと偽分布ペア
        ToDo: 出力：D_out
    """

    pseudo_docs_path: str = f"{OUTPUT_PATH}/pseudo_documents/2023-7-13/10-33-2"
    pseudo_docs_csv_list: List[str] = os.listdir(pseudo_docs_path)

    docs: List[List[str]] = []
    word_list: List[str] = []
    print("=== 文書の単語分割処理 ===")
    for docs_csv in tqdm(pseudo_docs_csv_list):
        words: List[str] = []
        df_pseudo: pl.DataFrame = data_loder(f"{pseudo_docs_path}/{docs_csv}", has_header=True)
        df_pseudo_filter = df_pseudo.filter((pl.col("speaker_name") != "議長") & (pl.col("speaker_name") != "副議長"))

        for utterance in tqdm(df_pseudo_filter["utterance"], leave=False):
            words += sudachi_tokenizer(utterance)
            word_list += words
        docs.append(words)

    # print(list(dict.fromkeys(word_list)))
    # print(len(list(dict.fromkeys(word_list))))
    word_list_del_dup: List[str] = list(dict.fromkeys(word_list))
    d_real: List[List[float]] = create_doc_representation(docs, word_list_del_dup)
    df_d_real: pl.DataFrame = pl.DataFrame(d_real, schema=word_list_del_dup)

    current_date, current_time = get_current_datetime()
    save_path: str = f"{OUTPUT_PATH}/pseudo_document_representation/{current_date}/{current_time}"
    make_dir(save_path)
    save_csv(df_d_real, save_path, "documents_representation.csv")


if __name__ == "__main__":
    main()
