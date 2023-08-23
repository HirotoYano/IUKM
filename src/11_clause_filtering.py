import os
import statistics
from typing import List

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
    data_path: str = f"{OUTPUT_PATH}/cos_similarity_to_com_unique_utterance/2023-8-19/14-14-23"
    file_name_list: List[str] = os.listdir(data_path)
    dim_list: List[int] = [3, 5, 10, 20, 30, 40, 50]

    current_date, current_time = get_current_datetime()
    save_path: str = (
        f"{OUTPUT_PATH}/clause_filtering/{current_date}/{current_time}"
    )
    make_dir(save_path)

    for i, file_name in enumerate(tqdm(file_name_list)):
        df_cos_sim_com: pl.DataFrame = data_loder(
            file_path=f"{data_path}/{file_name}", has_header=True
        )
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

        for cluster in tqdm(clusters, leave=False):
            df_cos_sim_com_cluster_filter: pl.DataFrame = (
                df_cos_sim_com.filter(pl.col("cluster") == cluster)
            )
            cos_similaritys: list = [
                cos_similarity
                for cos_similarity in df_cos_sim_com_cluster_filter[
                    "cos_similarity_to_com"
                ]
            ]
            mean: float = statistics.mean(cos_similaritys)
            df_assembly_add_cos_filtering = pl.concat(
                [
                    df_assembly_add_cos_filtering,
                    df_cos_sim_com_cluster_filter.filter(
                        pl.col("cos_similarity_to_com") > mean
                    ),
                ]
            )

        df_assembly_add_cos_filtering_sort: pl.DataFrame = (
            df_assembly_add_cos_filtering.select(
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
            df=df_assembly_add_cos_filtering_sort,
            path=save_path,
            file_name=f"assembly_cluster_cos_similarity_filtered_{dim_list[i]}_dimension.csv",
        )


if __name__ == "__main__":
    main()
