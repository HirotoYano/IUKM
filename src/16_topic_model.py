import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv
from codebase.dataprocessor.sudachi_tokenizer import sudachi_tokenizer

load_dotenv("/workspace/src/.env")
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
MINUTES_DATA_PATH: str = os.environ["MINUTES_DATA_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    discussion_path: str = f"{INTERIM_DATA_PATH}/discussion"
    # general_interpellation_list: List[str] = os.listdir(discussion_path)
    general_interpellation_list: List[str] = os.listdir(f"{OUTPUT_PATH}/compair_proposed_method")
    current_date, current_time = get_current_datetime()

    for general_interpellation in tqdm(general_interpellation_list):
        assembly_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/{general_interpellation}/assembly.csv"
        df_assembly: pl.DataFrame = data_loder(file_path=assembly_file_path, has_header=True).filter(
            (pl.col("speaker_name") != "議長") | (pl.col("speaker_name") != "副議長")
        )

        digest_file_path: str = f"{INTERIM_DATA_PATH}/{MINUTES_DATA_PATH}/{general_interpellation}/digest.csv"
        df_digest: pl.DataFrame = data_loder(file_path=digest_file_path, has_header=True).filter(
            (pl.col("label") == "policy.title") | (pl.col("label") == "reply.utterance")
        )

        short_summary_list: List[str] = []
        questioner_list: List[str] = []
        respondent_list: List[str] = []

        for i in tqdm(range(len(df_digest)), leave=False):
            if df_digest["label"][i] == "policy.title":
                short_summary_list.append(df_digest["utterance"][i])
                questioner_list.append(df_digest["speaker_name"][i])
            elif df_digest["label"][i] == "reply.utterance" and df_digest["label"][i - 1] == "policy.title":
                respondent_list.append(df_digest["speaker_name"][i])

        df_questioner_to_respondent = pl.DataFrame(
            {
                "short_summary": short_summary_list,
                "questioner": questioner_list,
                "respondent": respondent_list,
            },
            {
                "short_summary": pl.Utf8,
                "questioner": pl.Utf8,
                "respondent": pl.Utf8,
            },
        )

        discussion_file_path: str = f"{discussion_path}/{general_interpellation}"
        discussion_file_list: List[str] = os.listdir(discussion_file_path)

        docs: List[List[str]] = []
        for discussion_file in tqdm(discussion_file_list, leave=False):
            df_discussion: pl.DataFrame = data_loder(
                file_path=f"{discussion_file_path}/{discussion_file}", has_header=True
            )
            questioner = df_discussion["speaker_name"][0]
            df_questioner_to_respondent_filter: pl.DataFrame = df_questioner_to_respondent.filter(
                pl.col("questioner") == questioner
            )

            for i in tqdm(range(len(df_questioner_to_respondent_filter)), leave=False):
                doc: List[str] = []
                df_discussion_filter: pl.DataFrame = df_discussion.filter(
                    (pl.col("speaker_name") == df_questioner_to_respondent_filter["questioner"][i])
                    | (pl.col("speaker_name") == df_questioner_to_respondent_filter["respondent"][i])
                )

                for utterance in tqdm(df_discussion_filter["utterance"], leave=False):
                    doc.extend(sudachi_tokenizer(utterance))

                docs.append(doc)

        dictionary = Dictionary(docs)
        x = 1
        y = 0.1
        dictionary.filter_extremes(no_below=x, no_above=y)
        corpus = [dictionary.doc2bow(text) for text in docs]
        print(len(dictionary))

        num_topics = len(df_digest.filter(pl.col("label") == "policy.title"))
        lda = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, alpha=0.01)

        df: pl.DataFrame = pl.DataFrame()
        for t in tqdm(range(num_topics)):
            word = []
            for i, prob in lda.get_topic_terms(t, topn=len(dictionary)):
                word.append(dictionary.id2token[int(i)])

            _: pl.DataFrame = pl.DataFrame({f"cluster{t+1}": word}, {f"cluster{t+1}": pl.Utf8})
            df = df.with_columns(_)

        topic_model_result_save_path: str = f"{OUTPUT_PATH}/topic_model/{current_date}/{current_time}"
        make_dir(topic_model_result_save_path)
        save_csv(df=df, path=topic_model_result_save_path, file_name=f"{general_interpellation}.csv")


if __name__ == "__main__":
    main()
