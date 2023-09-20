import os
from typing import List

import polars as pl
from dotenv import load_dotenv

from codebase.dataprocessor.data_loder import data_loder

load_dotenv("/workspace/src/.env")
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    eval_path: str = f"{OUTPUT_PATH}/compair_proposed_method"
    digest_file_path: str = f"{INTERIM_DATA_PATH}/speaker_utterance_dataset"
    general_interpellation_list: List[str] = os.listdir(eval_path)

    print("match count")

    for general_interpellation in general_interpellation_list:
        df_digest: pl.DataFrame = data_loder(
            file_path=f"{digest_file_path}/{general_interpellation}/digest.csv", has_header=True
        ).filter((pl.col("label") == "policy.title") | (pl.col("label") == "reply.utterance"))

        df_debater_pairs: pl.DataFrame = pl.DataFrame(
            {
                "short_summary": [],
                "questioner": [],
                "respondent": [],
            },
            {
                "short_summary": pl.Utf8,
                "questioner": pl.Utf8,
                "respondent": pl.Utf8,
            },
        )

        for i in range(len(df_digest)):
            if df_digest["label"][i] == "policy.title":
                questioner = df_digest["speaker_name"][i]
                short_summary = df_digest["utterance"][i]
            elif df_digest["label"][i] == "reply.utterance":
                respondent: str = df_digest["speaker_name"][i]

                df_debater_pair: pl.DataFrame = pl.DataFrame(
                    {
                        "short_summary": short_summary,
                        "questioner": questioner,
                        "respondent": respondent,
                    },
                    {
                        "short_summary": pl.Utf8,
                        "questioner": pl.Utf8,
                        "respondent": pl.Utf8,
                    },
                )

                df_debater_pairs = pl.concat([df_debater_pairs, df_debater_pair])

        general_interpellation_path: str = f"{eval_path}/{general_interpellation}"
        discussion_dir_list: List[str] = os.listdir(general_interpellation_path)
        count: int = 0

        for discussion_dir in discussion_dir_list:
            qa_file_path: str = f"{general_interpellation_path}/{discussion_dir}"

            for i in range(len(df_debater_pairs)):
                qa_file: str = (
                    f"{qa_file_path}/"
                    + df_debater_pairs["questioner"][i]
                    + "_"
                    + df_debater_pairs["respondent"][i]
                    + ".csv"
                )
                is_file = os.path.isfile(qa_file)

                if is_file:
                    df_qa: pl.DataFrame = data_loder(file_path=f"{qa_file}", has_header=True)
                    words = df_qa.select(pl.col("word").sort_by("tf-idf", descending=True))[0:5]["word"].to_list()
                    print(f"{os.path.splitext(os.path.basename(qa_file))[0]}: {words}")

                    for word in words:
                        if word in df_debater_pairs["short_summary"][i]:
                            count += 1

        print(f"{general_interpellation}: {count}\n")


if __name__ == "__main__":
    main()
