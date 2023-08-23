import os
from copy import deepcopy
from random import shuffle
from typing import List

import polars as pl
from dotenv import load_dotenv

from codebase.dataprocessor.data_loder import data_loder

load_dotenv("/workspace/src/.env")
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    digest_data_path: str = f"{INTERIM_DATA_PATH}/speaker_utterance_dataset/一般質問(要旨)2月13日/digest.csv"
    short_summary_list: List[str] = (
        data_loder(file_path=digest_data_path, has_header=True)
        .filter(pl.col("label") == "policy.title")["utterance"]
        .to_list()
    )

    tf_idf_result_data_path: str = (
        f"{OUTPUT_PATH}/tf_idf_after_filtering/2023-8-22/11-49-21/top_five_words/top_five_words_10_dimension.csv"
    )
    df_tf_idf_result: pl.DataFrame = data_loder(file_path=tf_idf_result_data_path, has_header=True)

    topic_model_result_data_path: str = f"{OUTPUT_PATH}/topic_model/2023-8-22/9-19-13/result.csv"
    # topic_model_result_data_path: str = f"{OUTPUT_PATH}/topic_model/2023-8-22/9-14-36/result.csv"
    # topic_model_result_data_path: str = f"{OUTPUT_PATH}/topic_model/2023-8-22/9-9-13/result.csv"
    # topic_model_result_data_path: str = f"{OUTPUT_PATH}/topic_model/2023-8-22/8-56-7/result.csv"
    df_topic_model_result: pl.DataFrame = data_loder(file_path=topic_model_result_data_path, has_header=True)

    cluster_list: List[str] = df_tf_idf_result.columns
    shuffle(cluster_list)

    short_summary_list_tf_idf: List[str] = deepcopy(short_summary_list)
    short_summary_list_topic_model: List[str] = deepcopy(short_summary_list)

    counter_tf_idf: int = 0
    counter_topic_model: int = 0

    print("=== 提案手法 ===")
    for i in range(len(df_tf_idf_result)):
        for short_summary in short_summary_list:
            for cluster in cluster_list:
                if df_tf_idf_result[cluster][i] in short_summary:
                    print(short_summary)
                    print(df_tf_idf_result[cluster].to_list())
                    short_summary_list_tf_idf.remove(short_summary)
                    cluster_list.remove(cluster)
                    counter_tf_idf += 1
                    break

            if short_summary not in short_summary_list_tf_idf:
                short_summary_list = deepcopy(short_summary_list_tf_idf)

    print("\n=== LDA ===")
    for i in range(len(df_topic_model_result)):
        for short_summary in short_summary_list_topic_model:
            for cluster in cluster_list:
                if df_topic_model_result[cluster][i] in short_summary:
                    print(short_summary)
                    print(df_topic_model_result[cluster].to_list())
                    short_summary_list_topic_model.remove(short_summary)
                    cluster_list.remove(cluster)
                    counter_topic_model += 1
                    break

            if short_summary not in short_summary_list_topic_model:
                short_summary_list = deepcopy(short_summary_list_topic_model)

    print("\nmatch short summary count")
    print(f"proposed method: {counter_tf_idf}")
    print(f"LDA: {counter_topic_model}")


if __name__ == "__main__":
    main()
