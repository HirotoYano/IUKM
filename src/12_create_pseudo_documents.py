import os

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv

load_dotenv("/workspace/src/.env")
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    assembly_cluster_file_path: str = f"{OUTPUT_PATH}/dendrogram/2023-7-12/3-32-45/assembly_cluster_10_dimension.csv"
    df_cluster: pl.DataFrame = data_loder(file_path=assembly_cluster_file_path, has_header=True)
    num_clusters: int = len(df_cluster["cluster"].unique())

    current_date, current_time = get_current_datetime()
    save_path: str = f"{OUTPUT_PATH}/pseudo_documents/{current_date}/{current_time}"
    make_dir(save_path)

    for i in tqdm(range(1, num_clusters + 1)):
        df_cluster_filtered = df_cluster.filter(pl.col("cluster") == i)
        save_csv(df_cluster_filtered, save_path, f"cluster{i}.csv")


if __name__ == "__main__":
    main()
