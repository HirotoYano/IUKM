import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv
from codebase.dataprocessor.sudachi_tokenizer import sudachi_tokenizer

load_dotenv("/workspace/src/.env")
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    # フィルタリング前
    # assembly_file_path: str = f"{OUTPUT_PATH}/dendrogram/2023-6-11/14-48-45/assembly_cluster.csv"
    # フィルタリング後
    # assembly_file_path: str = f"{OUTPUT_PATH}/clause_filtering/2023-8-19/14-39-3"
    assembly_file_path: str = f"{OUTPUT_PATH}/clause_filtering_v2/2023-8-28/10-32-57"
    general_interpellation_list: List[str] = os.listdir(assembly_file_path)
    current_date, current_time = get_current_datetime()

    for general_interpellation in general_interpellation_list:
        assembly_cluster_csv_path: str = f"{assembly_file_path}/{general_interpellation}"
        assembly_cluster_csv_list: List[str] = os.listdir(assembly_cluster_csv_path)
        # dim_list: List[int] = [3, 5, 10, 20, 30, 40, 50]
        dim_list: List[int] = [10]

        # フィルタリング前
        # save_path: str = f"{OUTPUT_PATH}/tf_idf/{current_date}/{current_time}"
        # make_dir(save_path)

        # フィルタリング後
        save_path: str = (
            f"{OUTPUT_PATH}/tf_idf_after_filtering_v2/{current_date}/{current_time}/{general_interpellation}"
        )
        make_dir(f"{save_path}/score")
        make_dir(f"{save_path}/top_five_words")

        for i, assembly_cluster_csv in enumerate(tqdm(assembly_cluster_csv_list)):
            df_assembly = data_loder(
                file_path=f"{assembly_cluster_csv_path}/{assembly_cluster_csv}",
                has_header=True,
            )
            df_assembly_filter = df_assembly.filter(
                (pl.col("speaker_name") != "議長") & (pl.col("speaker_name") != "副議長")
            )
            clusters_list = df_assembly_filter["cluster"].unique().to_list()

            doc = []
            for cluster in tqdm(clusters_list, leave=False):
                word_list = []
                df_assembly_cluster_fiter = df_assembly_filter.filter(pl.col("cluster") == cluster)

                for utterance in df_assembly_cluster_fiter["utterance"]:
                    word_list += sudachi_tokenizer(utterance)
                doc.append(" ".join(word_list))

            vectorizer = TfidfVectorizer(smooth_idf=False)
            values = vectorizer.fit_transform(doc).toarray()
            words = vectorizer.get_feature_names_out().tolist()

            df_tf_idf = pl.DataFrame({"words": words})

            for j, value in enumerate(tqdm(values, leave=False), start=1):
                df_value = pl.DataFrame({f"cluster{j}": value})
                df_tf_idf = df_tf_idf.with_columns(df_value)

            df_result_words: pl.DataFrame = pl.DataFrame()

            for cluster in df_tf_idf.columns[1:]:
                words = df_tf_idf.select(pl.col("words", f"{cluster}").sort_by(f"{cluster}", descending=True))[
                    "words"
                ].to_list()

                df: pl.DataFrame = pl.DataFrame({f"{cluster}": words})
                df_result_words = df_result_words.with_columns(df)

            # save_csv(df=df_tf_idf, path=save_path, file_name="tf_idf.csv")
            save_csv(
                df=df_tf_idf,
                path=f"{save_path}/score",
                file_name=f"tf_idf_after_filtering_{dim_list[i]}_dimension.csv",
            )
            save_csv(
                df=df_result_words,
                path=f"{save_path}/top_five_words",
                file_name=f"top_five_words_{dim_list[i]}_dimension.csv",
            )


if __name__ == "__main__":
    main()
