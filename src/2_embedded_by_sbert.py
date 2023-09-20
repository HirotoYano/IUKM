import os
from typing import List

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir

load_dotenv("/workspace/src/.env")
LOG_PATH: str = os.environ["LOG_PATH"]
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
MINUTES_DATA_PATH: str = os.environ["MINUTES_DATA_PATH"]


def main():
    model_path: str = f"{LOG_PATH}/sentence_bert/2023-08-28"
    # general_interpellation_list: List[str] = os.listdir(model_path)

    # dim_list: List[int] = [3, 5, 10, 20, 30, 40, 50]
    dim_list: List[int] = [10]
    current_date, current_time = get_current_datetime()

    # for general_interpellation in tqdm(general_interpellation_list):
    # assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/{general_interpellation}/assembly.csv"
    assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/一般質問（要旨）2月26日/assembly.csv"
    df_assembly: pl.DataFrame = data_loder(file_path=assembly_file_path, has_header=True)

    each_dim_model_path_list: List[str] = os.listdir(model_path)

    # for i, dim_model_path in enumerate(tqdm(each_dim_model_path_list, leave=False)):
    # model: SentenceTransformer = SentenceTransformer(f"{model_path}/{dim_model_path}/{general_interpellation}")
    model: SentenceTransformer = SentenceTransformer(f"{model_path}/11-23-44/一般質問（要旨）2月26日")

    embedding_list: list = []
    for utterance in tqdm(df_assembly["utterance"], leave=False):
        embedding_list.append(model.encode(utterance))

    # save_path: str = f"{LOG_PATH}/embeddings_v3/{current_date}/{current_time}/{general_interpellation}"
    save_path: str = f"{LOG_PATH}/embeddings_v3/{current_date}/{current_time}/一般質問（要旨）2月26日"
    make_dir(save_path)

    np.savetxt(
        # f"{save_path}/vectors_{dim_list[i]}_dimension.csv",
        f"{save_path}/vectors_{dim_list[0]}_dimension.csv",
        embedding_list,
        delimiter=",",
    )


if __name__ == "__main__":
    main()
