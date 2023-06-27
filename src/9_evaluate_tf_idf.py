import os
import random

import polars as pl
from dotenv import load_dotenv

from codebase.dataprocessor.data_loder import data_loder

load_dotenv("/workspace/src/.env")
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]


def main():
    # フィルタリング前
    # data_path: str = f"{OUTPUT_PATH}/tf_idf/2023-6-12/1-13-26/tf_idf.csv"
    # フィルタリング後
    data_path: str = f"{OUTPUT_PATH}/tf_idf_after_filtering/2023-6-14/9-11-13/tf_idf_after_filtering.csv"

    df_tf_idf = data_loder(data_path, has_header=True)

    digest_data_path = f"{INTERIM_DATA_PATH}/speaker_utterance_dataset/一般質問(要旨)2月13日/digest.csv"
    df_digest = data_loder(digest_data_path, has_header=True)
    topics: list = df_digest.filter(pl.col("label") == "policy.title")["utterance"].to_list()
    topic_num: int = len(topics)

    clusters: list = df_tf_idf.columns[1:]
    # random.shuffle(clusters)

    match_count: int = 0
    match_words: list = []
    match_topics: list = []

    for cluster in clusters:
        words = df_tf_idf.select(pl.col("words", f"{cluster}").sort_by(f"{cluster}", descending=True))["words"][
            0:5
        ].to_list()

        for topic in topics:
            flag: int = 0
            for word in words:
                if word in topic:
                    match_words.append(word)
                    match_topics.append(topic)
                    match_count += 1
                    flag = 1
                    topics.remove(topic)
                    break
            if flag == 1:
                break

    print(f"match topic length: {len(match_topics)}")
    print(f"match topic: {match_topics}\n")
    print(f"unmatch topic length: {len(topics)}")
    print(f"unmatch topic: {topics}\n")
    print(f"match word length: {len(match_words)}")
    print(f"match word: {match_words}\n")
    print(f"match rate: {match_count / topic_num}")


if __name__ == "__main__":
    main()
