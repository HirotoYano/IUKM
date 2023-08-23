import os
from typing import List

import numpy as np
import polars as pl
from dotenv import load_dotenv
from numba import njit
from tqdm import tqdm

from codebase.dataprocessor.calculate_similarity import cos_sim
from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv

load_dotenv("/workspace/src/.env")
LOG_PATH: str = os.environ["LOG_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


@njit(nopython=False)
def calc_com(pos):
    return np.sum(pos, axis=0) / len(pos)


def main():
    vectors_path: str = f"{LOG_PATH}/embeddings/2023-6-10/13-6-29/vectors.csv"
    vectors: np.ndarray = np.loadtxt(vectors_path, delimiter=",")

    assembly_cluster_file_path: str = (
        f"{OUTPUT_PATH}/dendrogram/2023-7-12/3-32-45"
    )
    assembly_cluster_csv_list: List[str] = os.listdir(
        assembly_cluster_file_path
    )
    dim_list: list = [3, 5, 10, 20, 30, 40, 50]

    # assembly_cluster_path: str = (
    #     f"{OUTPUT_PATH}/dendrogram/2023-6-11/14-48-45/assembly_cluster.csv"
    # )
    current_date, current_time = get_current_datetime()
    # 通常
    # save_path: str = f"{OUTPUT_PATH}/cos_similarity_to_com/{current_date}/{current_time}"
    # フィルタリング用
    save_path: str = f"{OUTPUT_PATH}/cos_similarity_to_com_unique_utterance/{current_date}/{current_time}"
    make_dir(save_path)

    for i, assembly_cluster_path in enumerate(tqdm(assembly_cluster_csv_list)):
        df_assembly_add_cluster: pl.DataFrame = data_loder(
            file_path=f"{assembly_cluster_file_path}/{assembly_cluster_path}",
            has_header=True,
        )
        clusters: list = df_assembly_add_cluster["cluster"].unique().to_list()

        df_assembly_add_cos_simlarity: pl.DataFrame = pl.DataFrame(
            {
                "id": [],
                "speaker_name": [],
                "utterance": [],
                "label": [],
                "cluster": [],
                "cos_similarity_to_com": [],
            },
            {
                "id": pl.Int64,
                "speaker_name": pl.Utf8,
                "utterance": pl.Utf8,
                "label": pl.Int64,
                "cluster": pl.Int64,
                "cos_similarity_to_com": pl.Float64,
            },
        )

        for cluster in tqdm(clusters, leave=False):
            # 通常
            # df_assembly_add_cluster_filter: pl.DataFrame = df_assembly_add_cluster.filter(pl.col("cluster") == cluster)
            # フィルタリング用
            df_assembly_add_cluster_filter: pl.DataFrame = (
                df_assembly_add_cluster.filter(
                    pl.col("cluster") == cluster
                ).unique(subset=["utterance"])
            )

            vectors_per_cluster: list = np.array(
                [vectors[id] for id in df_assembly_add_cluster_filter["id"]]
            )

            center_of_mass: np.ndarray = calc_com(vectors_per_cluster)

            for j, vector_per_cluster in enumerate(
                tqdm(vectors_per_cluster, leave=False)
            ):
                cos_similarity_to_com = cos_sim(
                    center_of_mass, vector_per_cluster
                )
                new_seires = pl.DataFrame(
                    {"cos_similarity_to_com": cos_similarity_to_com}
                )
                df: pl.DataFrame = df_assembly_add_cluster_filter[
                    j
                ].with_columns(new_seires)
                df_assembly_add_cos_simlarity = pl.concat(
                    [df_assembly_add_cos_simlarity, df]
                )

        df_assembly_add_cos_simlarity_sort: pl.DataFrame = (
            df_assembly_add_cos_simlarity.select(
                pl.col(
                    "id",
                    "speaker_name",
                    "utterance",
                    "label",
                    "cluster",
                    "cos_similarity_to_com",
                ).sort_by("id")
            )
        )

        save_csv(
            df=df_assembly_add_cos_simlarity_sort,
            path=save_path,
            # file_name="cos_similarity_between_utterance_vector_and_com.csv",  # 通常
            file_name=f"cos_similarity_between_unique_utterance_vector_and_com_{dim_list[i]}_dimension.csv",  # フィルタリング用
        )


if __name__ == "__main__":
    main()
