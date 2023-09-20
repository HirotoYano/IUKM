import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv

# 環境変数の読み込み
load_dotenv("/workspace/src/.env")
LOG_PATH: str = os.environ["LOG_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
MINUTES_DATA_PATH: str = os.environ["MINUTES_DATA_PATH"]


def main():
    vectors_path: str = f"{LOG_PATH}/embeddings_v3/2023-8-29/6-45-59"
    vectors_csv_list: List[str] = os.listdir(vectors_path)
    # dim_list: List[int] = [3, 5, 10, 20, 30, 40, 50]
    dim_list: List[int] = [10]
    general_interpellation_list: List[str] = os.listdir(vectors_path)
    current_date, current_time = get_current_datetime()

    for general_interpellation in tqdm(general_interpellation_list):
        assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/{general_interpellation}/assembly.csv"
        df_assembly: pl.DataFrame = data_loder(file_path=assembly_file_path, has_header=True)

        save_path: str = f"{OUTPUT_PATH}/dendrogram_v3/{current_date}/{current_time}/{general_interpellation}"
        make_dir(save_path)

        for i, vectors_csv in enumerate(tqdm(vectors_csv_list, leave=False)):
            vectors: np.ndarray = np.loadtxt(f"{vectors_path}/{vectors_csv}/vectors_10_dimension.csv", delimiter=",")

            z = linkage(vectors, method="ward", metric="euclidean")

            # fig, ax = plt.subplots(figsize=(20, 5))
            # ax = dendrogram(z)

            # fig.savefig(f"{save_path}/{dim_list[i]}_dimension.png")

            clusters = fcluster(z, t=49, criterion="maxclust")
            cluster_series = pl.DataFrame(clusters).rename({"column_0": "cluster"})

            df_assembly_add_cluster = df_assembly.with_column(cluster_series)

            save_csv(
                df=df_assembly_add_cluster,
                path=save_path,
                file_name=f"assembly_cluster_{dim_list[i]}_dimension.csv",
            )


if __name__ == "__main__":
    main()
