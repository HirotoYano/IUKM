import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
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
            word_list.extend(sudachi_tokenizer(utterance))
        doc.append(word_list)

    dictionary = Dictionary(doc)
    x = 1
    y = 0.1
    dictionary.filter_extremes(no_below=x, no_above=y)
    corpus = [dictionary.doc2bow(text) for text in doc]

    num_topics = 49
    lda = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, alpha=0.01)

    df: pl.DataFrame = pl.DataFrame()
    for t in tqdm(range(num_topics)):
        word = []
        for i, prob in lda.get_topic_terms(t, topn=5):
            word.append(dictionary.id2token[int(i)])

        _: pl.DataFrame = pl.DataFrame({f"cluster{t+1}": word}, {f"cluster{t+1}": pl.Utf8})
        df = df.with_columns(_)

    current_date, current_time = get_current_datetime()
    topic_model_result_save_path: str = f"{OUTPUT_PATH}/topic_model/{current_date}/{current_time}"
    make_dir(topic_model_result_save_path)
    save_csv(df=df, path=topic_model_result_save_path, file_name="result.csv")


if __name__ == "__main__":
    main()
