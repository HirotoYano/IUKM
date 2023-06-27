import os
import statistics

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
    # data_path: str = f"{OUTPUT_PATH}/cos_similarity_to_com/2023-6-13/16-46-48/"
    data_path: str = f"{OUTPUT_PATH}/cos_similarity_to_com_unique_utterance/2023-6-14/8-59-58/"
    file_name: str = "cos_similarity_between_unique_utterance_vector_and_com.csv"
    df_cos_sim_com: pl.DataFrame = data_loder(data_path + file_name, has_header=True)
    clusters: list = df_cos_sim_com["cluster"].unique().to_list()

    df_assembly_add_cos_filtering: pl.DataFrame = pl.DataFrame(
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

    for cluster in tqdm(clusters):
        df_cos_sim_com_cluster_filter: pl.DataFrame = df_cos_sim_com.filter(pl.col("cluster") == cluster)
        cos_similaritys: list = [
            cos_similarity for cos_similarity in df_cos_sim_com_cluster_filter["cos_similarity_to_com"]
        ]
        mean: float = statistics.mean(cos_similaritys)
        df_assembly_add_cos_filtering = pl.concat(
            [
                df_assembly_add_cos_filtering,
                df_cos_sim_com_cluster_filter.filter(pl.col("cos_similarity_to_com") > mean),
            ]
        )

    df_assembly_add_cos_filtering_sort: pl.DataFrame = df_assembly_add_cos_filtering.select(
        pl.col(
            "id",
            "speaker_name",
            "utterance",
            "label",
            "cluster",
            "cos_similarity_to_com",
        ).sort_by("id")
    )
    current_date, current_time = get_current_datetime()
    save_path: str = f"{OUTPUT_PATH}/clause_filtering/{current_date}/{current_time}"
    make_dir(save_path)
    save_csv(
        df=df_assembly_add_cos_filtering_sort,
        path=save_path,
        file_name="assembly_cluster_cos_similarity_filtered.csv",
    )


if __name__ == "__main__":
    main()
