import os
from random import randrange
from typing import List

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv

load_dotenv("/workspace/src/.env")
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    discussion_path: str = f"{INTERIM_DATA_PATH}/discussion"
    # general_interpellation_list: List[str] = os.listdir(discussion_path)
    # general_interpellation_list: List[str] = os.listdir(f"{OUTPUT_PATH}/compair_proposed_method")
    general_interpellation_list: List[str] = ["一般質問(要旨)2月13日"]

    for general_interpellation in tqdm(general_interpellation_list):
        general_interpellation_path: str = f"{discussion_path}/{general_interpellation}"
        file_name_list: List[str] = os.listdir(general_interpellation_path)
        anchor_clause_id: List[int] = []
        anchor_speaker_name: List[str] = []
        anchor_utterance: List[str] = []
        anchor_speaker_label: List[int] = []
        positive_clause_id: List[int] = []
        positive_speaker_name: List[str] = []
        positive_utterance: List[str] = []
        positive_speaker_label: List[int] = []
        other_clause_id: List[int] = []
        other_speaker_name: List[str] = []
        other_utterance: List[str] = []
        other_speaker_label: List[int] = []

        for file_name in tqdm(file_name_list, leave=False):
            df_discussion: pl.DataFrame = data_loder(
                file_path=f"{general_interpellation_path}/{file_name}", has_header=True
            ).with_columns([pl.col("utterance").apply(len).alias("len")])
            questioner: str = df_discussion[0]["speaker_name"][0]
            respondent_list: List[str] = (
                df_discussion.filter(pl.col("speaker_name") != questioner)["speaker_name"].unique().to_list()
            )
            df_quesioner = df_discussion.filter((pl.col("speaker_name") == questioner) & (pl.col("len") > 10))
            df_quesioner = df_quesioner.with_columns([pl.col("utterance").apply(len).alias("len")])
            for respondent in tqdm(respondent_list, leave=False):
                df_respondent = df_discussion.filter((pl.col("speaker_name") == respondent) & (pl.col("len") > 10))
                df_discussion_filter: pl.DataFrame = df_discussion.filter(
                    (pl.col("speaker_name") != questioner) & (pl.col("speaker_name") != respondent)
                )

                for i in tqdm(range(len(df_quesioner)), leave=False):
                    for j in tqdm(range(len(df_respondent)), leave=False):
                        num: int = randrange(len(df_discussion_filter))

                        anchor_clause_id.append(df_respondent["id"][j])
                        anchor_speaker_name.append(df_respondent["speaker_name"][j])
                        anchor_utterance.append(df_respondent["utterance"][j])
                        anchor_speaker_label.append(df_respondent["label"][j])
                        positive_clause_id.append(df_quesioner["id"][i])
                        positive_speaker_name.append(df_quesioner["speaker_name"][i])
                        positive_utterance.append(df_quesioner["utterance"][i])
                        positive_speaker_label.append(df_quesioner["label"][i])
                        other_clause_id.append(df_discussion_filter["id"][num])
                        other_speaker_name.append(df_discussion_filter["speaker_name"][num])
                        other_utterance.append(df_discussion_filter["utterance"][num])
                        other_speaker_label.append(df_discussion_filter["label"][num])

        df_result: pl.DataFrame = pl.DataFrame(
            {
                "anchor_clause_id": anchor_clause_id,
                "anchor_speaker_name": anchor_speaker_name,
                "anchor_utterance": anchor_utterance,
                "anchor_speaker_label": anchor_speaker_label,
                "positive_clause_id": positive_clause_id,
                "positive_speaker_name": positive_speaker_name,
                "positive_utterance": positive_utterance,
                "positive_speaker_label": positive_speaker_label,
                "other_clause_id": other_clause_id,
                "other_speaker_name": other_speaker_name,
                "other_utterance": other_utterance,
                "other_speaker_label": other_speaker_label,
            },
            {
                "anchor_clause_id": pl.Int64,
                "anchor_speaker_name": pl.Utf8,
                "anchor_utterance": pl.Utf8,
                "anchor_speaker_label": pl.Int64,
                "positive_clause_id": pl.Int64,
                "positive_speaker_name": pl.Utf8,
                "positive_utterance": pl.Utf8,
                "positive_speaker_label": pl.Int64,
                "other_clause_id": pl.Int64,
                "other_speaker_name": pl.Utf8,
                "other_utterance": pl.Utf8,
                "other_speaker_label": pl.Int64,
            },
        )

        save_path: str = f"{INTERIM_DATA_PATH}/adjacency_pair_v3"
        make_dir(save_path)
        save_csv(df=df_result, path=save_path, file_name=f"{general_interpellation}.csv")


if __name__ == "__main__":
    main()
