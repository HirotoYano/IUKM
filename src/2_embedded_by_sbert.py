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
    assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/一般質問(要旨)2月13日/assembly.csv"
    df_assembly: pl.DataFrame = data_loder(file_path=assembly_file_path, has_header=True)

    model_path: str = f"{LOG_PATH}/sentence_bert/2023-07-10"
    each_dim_model_path_list: List[str] = os.listdir(model_path)

    dim_list: List[int] = [3, 5, 10, 20, 30, 40, 50]

    current_date, current_time = get_current_datetime()
    save_path: str = f"{LOG_PATH}/embeddings/{current_date}/{current_time}"
    make_dir(save_path)

    for i, dim_model_path in enumerate(tqdm(each_dim_model_path_list)):
        model: SentenceTransformer = SentenceTransformer(f"{model_path}/{dim_model_path}/")

        embedding_list: list = []
        for utterance in tqdm(df_assembly["utterance"], leave=False):
            embedding_list.append(model.encode(utterance))

        np.savetxt(
            f"{save_path}/vectors_{dim_list[i]}_dimension.csv",
            embedding_list,
            delimiter=",",
        )


if __name__ == "__main__":
    main()
