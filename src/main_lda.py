# %%
# ライブラリをインポート
from __future__ import annotations

import os

import polars as pl
from dotenv import load_dotenv
from gensim.models import LdaModel, TfidfModel
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dict import make_dict
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv

# from codebase.dataprocessor.janome_tokenizer import janome_tokenizer
from codebase.dataprocessor.sudachi_tokenizer import sudachi_tokenizer

# %%
# 環境変数の読み込み
load_dotenv("/workspace/src/.env")
interim_data_path = os.environ["INTERIM_DATA_PATH"]
minutes_data_path = os.environ["MINUTES_DATA_PATH"]
log_path = os.environ["LOG_PATH"]


# %%
# 全議事録データに対してLDAを実行し、結果をCSVに保存する関数
def get_words_per_topic_in_lda(
    data_dir_list: list, current_date: str, current_time: str
):
    for data_dir in tqdm(data_dir_list):
        assembly_data_path: str = (
            f"{interim_data_path}{minutes_data_path}/{data_dir}/assembly.csv"
        )
        digest_data_path: str = (
            f"{interim_data_path}{minutes_data_path}/{data_dir}/digest.csv"
        )

        # データ読み込み
        df_assembly = data_loder(assembly_data_path, has_header=True)
        df_digest = data_loder(digest_data_path, has_header=True)

        # 形態素解析
        token_lists = []

        for clause in tqdm(df_assembly["utterance"]):
            if len(clause) > 5:
                # token_lists.append(janome_tokenizer(clause))
                token_lists.append(sudachi_tokenizer(clause))

        # 辞書作成
        dictionary, corpus = make_dict(token_lists)

        # TF-IDF
        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        # LDAを実行
        num_topics = len(df_digest.filter(pl.col("label") == "policy.title"))
        lda = LdaModel(
            # corpus=corpus, # BoWを使用する場合
            corpus=corpus_tfidf,  # TF-IDFを使用する場合
            id2word=dictionary,
            num_topics=num_topics,
            alpha=0.01,
        )

        # 結果を出力
        df_results = pl.DataFrame()

        for t in range(num_topics):
            word = []

            for i, _ in lda.get_topic_terms(t, topn=5):
                word.append(dictionary.id2token[int(i)])
            new_seires = pl.DataFrame({f"topic{t}": word})
            df_results = df_results.with_columns(new_seires)

        # 結果をCSVに保存
        save_path = f"{log_path}/lda_result/{current_date}/{current_time}"
        make_dir(path=save_path)
        save_csv(
            df=df_results,
            path=save_path,
            file_name=f"{data_dir}.csv",
        )


# %%
# 出力したトピック毎の単語と議会だよりの話題を比較
def compare_topics_and_words(result_csv_list: list) -> float:
    match_count: int = 0
    full_count: int = 0

    for result_csv in tqdm(result_csv_list):
        # データ読み込み
        df_result = data_loder(
            # f"{log_path}/lda_result/2023-5-31/7-14-21/{result_csv}", # BoW
            f"{log_path}/lda_result/2023-6-5/7-54-48/{result_csv}",  # TF-IDF
            has_header=True,
        )

        digest_dir: str = os.path.splitext(result_csv)[0]
        df_digest = data_loder(
            f"{interim_data_path}{minutes_data_path}/{digest_dir}/digest.csv",
            has_header=True,
        ).filter(pl.col("label") == "policy.title")

        for title in df_digest["utterance"]:
            full_count += 1
            flag: int = 0
            for topic in df_result.columns:
                for topic_word in df_result[topic]:
                    if topic_word in title:
                        match_count += 1
                        flag = 1
                        break
                if flag == 1:
                    break

    return match_count / full_count


# %%
# 各話題のマッチ数を算出
def get_number_of_matches_between_topics_and_words(
    result_csv_list: list, save_path: str, save_file_name: str
):
    df = pl.DataFrame(
        {
            "file_name": [],
            "topic": [],
            "match_num": [],
        },
        {
            "file_name": pl.Utf8,
            "topic": pl.Utf8,
            "match_num": pl.Int64,
        },
    )

    for result_csv in tqdm(result_csv_list):
        # データ読み込み
        df_result = data_loder(
            # f"{log_path}/lda_result/2023-5-31/7-14-21/{result_csv}", # BoW
            f"{log_path}/lda_result/2023-6-5/7-54-48/{result_csv}",  # TF-IDF
            has_header=True,
        )

        digest_dir: str = os.path.splitext(result_csv)[0]
        df_digest = data_loder(
            f"{interim_data_path}{minutes_data_path}/{digest_dir}/digest.csv",
            has_header=True,
        ).filter(pl.col("label") == "policy.title")

        for title in df_digest["utterance"]:
            match_count = 0

            for topic in df_result.columns:
                for topic_word in df_result[topic]:
                    if topic_word in title:
                        match_count += 1

            df_temp = pl.DataFrame(
                {
                    "file_name": digest_dir,
                    "topic": title,
                    "match_num": match_count,
                }
            )

            df = pl.concat([df, df_temp])

    save_csv(df=df, path=save_path, file_name=save_file_name)


# %%
# main関数
def main(lda_flag: bool = True):
    data_dir_list = os.listdir(f"{interim_data_path}{minutes_data_path}")
    current_date, current_time = get_current_datetime()

    if lda_flag is True:
        get_words_per_topic_in_lda(
            data_dir_list=data_dir_list,
            current_date=current_date,
            current_time=current_time,
        )

    # BoW
    # result_csv_list = os.listdir(f"{log_path}/lda_result/2023-5-31/7-14-21")
    # TF-IDF
    result_csv_list = os.listdir(f"{log_path}/lda_result/2023-6-5/7-54-48")
    coincident_ratio = compare_topics_and_words(
        result_csv_list=result_csv_list
    )
    print(coincident_ratio)

    save_file_path = (
        f"{log_path}/match_topic_to_word/{current_date}/{current_time}"
    )
    make_dir(path=save_file_path)
    get_number_of_matches_between_topics_and_words(
        result_csv_list=result_csv_list,
        save_path=save_file_path,
        save_file_name="test_tfidf.csv",
    )


# main関数の実行
if __name__ == "__main__":
    main(lda_flag=False)
    # main()
