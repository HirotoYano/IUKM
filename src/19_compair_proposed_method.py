import os
from typing import List

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv
from codebase.dataprocessor.sudachi_tokenizer import sudachi_tokenizer

load_dotenv("/workspace/src/.env")
INTERIM_DATA_PATH: str = os.environ["INTERIM_DATA_PATH"]
OUTPUT_PATH: str = os.environ["OUTPUT_PATH"]


def main():
    discussion_file_path: str = f"{INTERIM_DATA_PATH}/discussion"
    assembly_file_path: str = f"{INTERIM_DATA_PATH}/speaker_utterance_dataset"
    general_interpellation_list: List[str] = os.listdir(discussion_file_path)

    for general_interpellation in tqdm(general_interpellation_list):
        discussion_file_list: List[str] = os.listdir(f"{discussion_file_path}/{general_interpellation}")
        df_assembly: pl.DataFrame = data_loder(
            file_path=f"{assembly_file_path}/{general_interpellation}/assembly.csv", has_header=True
        ).filter((pl.col("speaker_name") != "議長") & (pl.col("speaker_name") != "副議長"))

        word_doc_freq_list: List[str] = [
            word
            for utterance in tqdm(df_assembly["utterance"], leave=False)
            for word in tqdm(sudachi_tokenizer(utterance), leave=False)
        ]

        df_doc_freq: pl.DataFrame = pl.DataFrame({"word": [], "count": []}, {"word": pl.Utf8, "count": pl.Int64})
        for word_doc_freq in tqdm(set(word_doc_freq_list), leave=False):
            df_word_count: pl.DataFrame = pl.DataFrame(
                {"word": word_doc_freq, "count": word_doc_freq_list.count(word_doc_freq)},
                {"word": pl.Utf8, "count": pl.Int64},
            )
            df_doc_freq = pl.concat([df_doc_freq, df_word_count])

        for i, discussion_file in enumerate(tqdm(discussion_file_list, leave=False)):
            df_discussion: pl.DataFrame = data_loder(
                file_path=f"{discussion_file_path}/{general_interpellation}/{discussion_file}", has_header=True
            )
            questioner: str = df_discussion[0]["speaker_name"][0]

            respondent_list: List[str] = (
                df_discussion.filter(pl.col("speaker_name") != questioner)["speaker_name"].unique().to_list()
            )

            for respondent in tqdm(respondent_list, leave=False):
                df_question_and_respond: pl.DataFrame = df_discussion.filter(
                    (pl.col("speaker_name") == questioner | (pl.col("speaker_name") == respondent))
                )

                word_term_freq_list: List[str] = [
                    word
                    for utterance in tqdm(df_question_and_respond["utterance"], leave=False)
                    for word in tqdm(sudachi_tokenizer(utterance), leave=False)
                ]

                df_result: pl.DataFrame = pl.DataFrame(
                    {"word": [], "tf-idf": []}, {"word": pl.Utf8, "tf-idf": pl.Float64}
                )
                for word_term_freq in tqdm(set(word_term_freq_list), leave=False):
                    df_term_div_doc: pl.DataFrame = pl.DataFrame(
                        {
                            "word": word_term_freq,
                            "tf-idf": word_term_freq_list.count(word_term_freq)
                            / df_doc_freq.filter(pl.col("word") == word_term_freq)["count"],
                        },
                        {"word": pl.Utf8, "tf-idf": pl.Float64},
                    )
                    df_result = pl.concat([df_result, df_term_div_doc])

                save_path: str = f"{OUTPUT_PATH}/compair_proposed_method/{general_interpellation}/discussion{i}"
                make_dir(path=save_path)
                save_csv(df=df_result, path=save_path, file_name=f"{questioner}_{respondent}.csv")


if __name__ == "__main__":
    main()
