import os

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir

# 環境変数の読み込み
load_dotenv("/workspace/src/.env")
LOG_PATH = os.environ["LOG_PATH"]
OUTPUT_PATH = os.environ["OUTPUT_PATH"]
INTERIM_DATA_PATH = os.environ["INTERIM_DATA_PATH"]
MINUTES_DATA_PATH = os.environ["MINUTES_DATA_PATH"]


def main():
    reduce_vector_path: str = (
        f"{LOG_PATH}/reduce_dimension/2023-6-10/13-21-42/reduced_vectors.csv"
    )
    cluster_label_path: str = (
        f"{LOG_PATH}/clustering/2023-6-10/14-6-48/cluster_label.csv"
    )
    vectors: np.ndarray = np.loadtxt(reduce_vector_path, delimiter=",")
    cluster_labels: np.ndarray = np.loadtxt(
        cluster_label_path, delimiter=",", dtype="int64"
    )

    # クラスタリングラベルで色分け
    # fig, ax = plt.subplots(figsize=(10, 8))
    # cluster_labels_unique = np.unique(cluster_labels)
    # for label in tqdm(cluster_labels_unique):
    #     array = np.empty((0, 2), float)
    #     if label != -1:
    #         for i, cluster_label in enumerate(cluster_labels):
    #             if label == cluster_label:
    #                 array = np.append(array, [vectors[i]], axis=0)
    #         ax.scatter(array[:, 0], array[:, 1], label=label)

    # 話者で色分け
    assembly_file_path: str = (
        f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/一般質問(要旨)2月13日/assembly.csv"
    )
    df_assembly = data_loder(file_path=assembly_file_path, has_header=True)
    speaker_list = df_assembly["speaker_name"].unique().to_list()
    # plt.rcParams["font.family"] = "MS Gothic"

    fig, ax = plt.subplots(figsize=(10, 8))
    for speaker in tqdm(speaker_list):
        array = np.empty((0, 2), float)
        for i, cluster_label in enumerate(cluster_labels):
            if (
                cluster_label != -1
                and speaker == df_assembly["speaker_name"][i]
            ):
                array = np.append(array, [vectors[i]], axis=0)
        ax.scatter(array[:, 0], array[:, 1], label=speaker)

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=11,
        ncol=2,
    )
    plt.grid()
    plt.show()

    current_date, current_time = get_current_datetime()
    save_path = f"{OUTPUT_PATH}/fig_clustering/{current_date}/{current_time}"
    make_dir(save_path)
    fig.savefig(f"{save_path}/test.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
