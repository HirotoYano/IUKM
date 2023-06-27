import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir

# 環境変数の読み込み
load_dotenv("/workspace/src/.env")
LOG_PATH = os.environ["LOG_PATH"]
INTERIM_DATA_PATH = os.environ["INTERIM_DATA_PATH"]
MINUTES_DATA_PATH = os.environ["MINUTES_DATA_PATH"]


def main():
    # 議会だよりデータの読み込み
    assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/一般質問(要旨)2月13日/assembly.csv"
    df_assembly = data_loder(file_path=assembly_file_path, has_header=True)

    # fine-tuning済みモデルの読み込み
    # 20 dimension fine-tuning model
    _20_dimension_model_path: str = f"{LOG_PATH}/sentence_bert/2023-06-23/06-40-10/"
    # 10 dimension fine-tuning model
    _10_dimension_model_path: str = f"{LOG_PATH}/sentence_bert/2023-06-10/09-53-33/"
    # 5 dimension fine-tuning model
    _5_dimension_model_path: str = f"{LOG_PATH}/sentence_bert/2023-06-23/04-26-07/"
    # 3 dimension fine-tuning model
    _3_dimension_model_path: str = f"{LOG_PATH}/sentence_bert/2023-06-23/05-03-59/"
    # 1 dimension fine-tuning model
    _1_dimension_model_path: str = f"{LOG_PATH}/sentence_bert/2023-06-23/06-12-40/"

    model_path_list: list = [
        _1_dimension_model_path,
        _3_dimension_model_path,
        _5_dimension_model_path,
        _10_dimension_model_path,
        _20_dimension_model_path,
    ]
    dim_list: list = [1, 3, 5, 10, 20]

    current_date, current_time = get_current_datetime()
    save_path = f"{LOG_PATH}/embeddings/{current_date}/{current_time}"
    make_dir(save_path)

    for i, model_path in enumerate(tqdm(model_path_list)):
        model: SentenceTransformer = SentenceTransformer(model_path)

        embedding_list = []
        for utterance in tqdm(df_assembly["utterance"], leave=False):
            embedding_list.append(model.encode(utterance))

        np.savetxt(f"{save_path}/vectors_{dim_list[i]}_dimension.csv", embedding_list, delimiter=",")


if __name__ == "__main__":
    main()
