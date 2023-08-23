import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv

load_dotenv("/workspace/src/.env")
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
MINUTES_DATA_PATH: str = os.environ["MINUTES_DATA_PATH"]


def main():
    assembly_cluster_file_path: str = (
        f"{OUTPUT_PATH}/dendrogram/2023-7-12/3-32-45"
    )
    assembly_cluster_csv_list: List[str] = os.listdir(
        assembly_cluster_file_path
    )
    dim_list: list = [3, 5, 10, 20, 30, 40, 50]

    adjacency_file_path: str = (
        f"{INTERIM_DATA_PATH}/adjacency_pair/一般質問(要旨)2月13日.csv"
    )
    df_adjacency = data_loder(file_path=adjacency_file_path, has_header=True)
    df_adjacency_select = df_adjacency.select(
        ["anchor_clause_id", "positive_clause_id"]
    )

    df_accuracy = pl.DataFrame({"accuracy": 0})

    current_date, current_time = get_current_datetime()
    save_path: str = (
        f"{OUTPUT_PATH}/word_clustering/{current_date}/{current_time}"
    )
    make_dir(save_path)

    for i, assembly_cluster_csv in enumerate(tqdm(assembly_cluster_csv_list)):
        df_assembly = data_loder(
            file_path=f"{assembly_cluster_file_path}/{assembly_cluster_csv}",
            has_header=True,
        )
        df_assembly_filter = df_assembly.filter(
            (pl.col("speaker_name") != "議長")
            & (pl.col("speaker_name") != "副議長")
        )

        select_num = 0
        all_num = 0
        for j in tqdm(range(len(df_adjacency_select)), leave=False):
            cluster_id_anchor = df_assembly_filter.filter(
                pl.col("id") == df_adjacency_select["anchor_clause_id"][j]
            )["cluster"][0]
            cluster_id_positive = df_assembly_filter.filter(
                pl.col("id") == df_adjacency_select["positive_clause_id"][j]
            )["cluster"][0]

            if cluster_id_anchor == cluster_id_positive:
                select_num += 1

            all_num += 1

        df = pl.DataFrame(
            {f"accuracy_{dim_list[i]}_dimension": select_num / all_num}
        )
        df_accuracy = df_accuracy.with_columns(df)

    df_accuracy = df_accuracy.drop("accuracy")
    print(df_accuracy)
    # save_csv(df=df_accuracy, path=save_path, file_name="accuracy.csv")


if __name__ == "__main__":
    main()
