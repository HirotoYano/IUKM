import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv
from codebase.dataprocessor.sudachi_tokenizer import sudachi_tokenizer

load_dotenv("/workspace/src/.env")
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
MINUTES_DATA_PATH: str = os.environ["MINUTES_DATA_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/一般質問(要旨)2月13日/assembly.csv"
    df_assembly: pl.DataFrame = data_loder(file_path=assembly_file_path, has_header=True)
    division_id: List[int] = df_assembly.filter(
        ((pl.col("speaker_name") == "議長") | (pl.col("speaker_name") == "副議長"))
        & (pl.col("utterance").str.contains(r"[一二三四五六七八九十百]+番.+"))
    )["id"].to_list()
    division_id.append(df_assembly[-1]["id"][0])

    discussion_csv_save_path: str = f"{OUTPUT_PATH}/discussion"
    make_dir(discussion_csv_save_path)

    doc = []
    for i in tqdm(range(len(division_id) - 1)):
        df_assembly_filter: pl.DataFrame = df_assembly[division_id[i] : division_id[i + 1]].filter(
            (pl.col("speaker_name") != "議長") & (pl.col("speaker_name") != "副議長")
        )
        save_csv(df=df_assembly_filter, path=discussion_csv_save_path, file_name=f"{i}.csv")

        word_list = []
        for utterance in tqdm(df_assembly_filter["utterance"], leave=False):
            word_list += sudachi_tokenizer(utterance)
        doc.append(" ".join(word_list))

    print(doc)

    vectorizer = TfidfVectorizer(smooth_idf=False)
    values = vectorizer.fit_transform(doc).toarray()
    words = vectorizer.get_feature_names_out().tolist()

    df_tf_idf = pl.DataFrame({"words": words})

    for j, value in enumerate(tqdm(values, leave=False), start=1):
        df_value = pl.DataFrame({f"cluster{j}": value})
        df_tf_idf = df_tf_idf.with_columns(df_value)

    print(df_tf_idf)


if __name__ == "__main__":
    main()
