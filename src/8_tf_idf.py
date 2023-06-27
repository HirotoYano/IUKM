import os

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
    assembly_file_path: str = (
        f"{OUTPUT_PATH}/clause_filtering/2023-6-14/9-7-40/assembly_cluster_cos_similarity_filtered.csv"
    )
    df_assembly = data_loder(file_path=assembly_file_path, has_header=True)
    df_assembly_filter = df_assembly.filter((pl.col("speaker_name") != "議長") & (pl.col("speaker_name") != "副議長"))
    clusters_list = df_assembly_filter["cluster"].unique().to_list()

    doc = []
    for cluster in tqdm(clusters_list):
        word_list = []
        df_assembly_cluster_fiter = df_assembly_filter.filter(pl.col("cluster") == cluster)

        for utterance in df_assembly_cluster_fiter["utterance"]:
            word_list += sudachi_tokenizer(utterance)
        doc.append(" ".join(word_list))

    vectorizer = TfidfVectorizer(smooth_idf=False)
    values = vectorizer.fit_transform(doc).toarray()
    words = vectorizer.get_feature_names_out().tolist()

    df_tf_idf = pl.DataFrame({"words": words})

    for i, value in enumerate(values, start=1):
        df_value = pl.DataFrame({f"cluster{i}": value})
        df_tf_idf = df_tf_idf.with_columns(df_value)

    current_date, current_time = get_current_datetime()
    # フィルタリング前
    # save_path: str = f"{OUTPUT_PATH}/tf_idf/{current_date}/{current_time}"
    # make_dir(save_path)
    # save_csv(df=df_tf_idf, path=save_path, file_name="tf_idf.csv")

    # フィルタリング後
    save_path: str = f"{OUTPUT_PATH}/tf_idf_after_filtering/{current_date}/{current_time}"
    make_dir(save_path)
    save_csv(df=df_tf_idf, path=save_path, file_name="tf_idf_after_filtering.csv")


if __name__ == "__main__":
    main()
