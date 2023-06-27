import os

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
    # load vectors
    # 20 dimensional vectors
    _20_dimensional_vectors_path: str = f"{LOG_PATH}/embeddings/2023-6-23/7-37-34/vectors_20_dimension.csv"
    # 10 dimensional vectors
    _10_dimensional_vectors_path: str = f"{LOG_PATH}/embeddings/2023-6-23/7-37-34/vectors_10_dimension.csv"
    # 5 dimensional vectors
    _5_dimensional_vectors_path: str = f"{LOG_PATH}/embeddings/2023-6-23/7-37-34/vectors_5_dimension.csv"
    # 3 dimensional vectors
    _3_dimensional_vectors_path: str = f"{LOG_PATH}/embeddings/2023-6-23/7-37-34/vectors_3_dimension.csv"

    vectors_path_list: list = [
        _3_dimensional_vectors_path,
        _5_dimensional_vectors_path,
        _10_dimensional_vectors_path,
        _20_dimensional_vectors_path,
    ]
    dim_list: list = [3, 5, 10, 20]

    current_date, current_time = get_current_datetime()
    save_path: str = f"{OUTPUT_PATH}/dendrogram/{current_date}/{current_time}"
    make_dir(save_path)

    for i, vectors_path in enumerate(tqdm(vectors_path_list)):
        vectors: np.ndarray = np.loadtxt(vectors_path, delimiter=",")

        z = linkage(vectors, method="ward", metric="euclidean")
        # df = pl.DataFrame(z)

        fig, ax = plt.subplots(figsize=(20, 5))
        ax = dendrogram(z)

        fig.savefig(f"{save_path}/{dim_list[i]}_dimension.png")

        clusters = fcluster(z, t=49, criterion="maxclust")
        assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/一般質問(要旨)2月13日/assembly.csv"
        df_assembly = data_loder(file_path=assembly_file_path, has_header=True)
        cluster_series = pl.DataFrame(clusters).rename({"column_0": "cluster"})

        df_assembly_add_cluster = df_assembly.with_column(cluster_series)

        save_csv(
            df=df_assembly_add_cluster,
            path=save_path,
            file_name=f"assembly_cluster_{dim_list[i]}_dimension.csv",
        )


if __name__ == "__main__":
    main()
