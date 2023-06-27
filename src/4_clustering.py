import os

import hdbscan
import numpy as np
from dotenv import load_dotenv

from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir

# 環境変数の読み込み
load_dotenv("/workspace/src/.env")
LOG_PATH: str = os.environ["LOG_PATH"]


def main():
    # load vectors
    vectors_path: str = f"{LOG_PATH}/embeddings/2023-6-10/13-6-29/vectors.csv"
    vectors: np.ndarray = np.loadtxt(vectors_path, delimiter=",")

    # clustering
    clusterer: hdbscan.HDBSCAN = hdbscan.HDBSCAN(
        gen_min_span_tree=True, min_cluster_size=15
    )
    cluster_labels: np.ndarray = clusterer.fit_predict(vectors)

    # output results
    current_date, current_time = get_current_datetime()
    save_path: str = f"{LOG_PATH}/clustering/{current_date}/{current_time}"
    make_dir(save_path)
    np.savetxt(f"{save_path}/cluster_label.csv", cluster_labels, delimiter=",")


if __name__ == "__main__":
    main()
