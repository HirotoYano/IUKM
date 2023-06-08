# %%
import itertools
import os

import more_itertools
import polars as pl
from tqdm import tqdm


# %%
def make_triple_pair_process(_df, spe_label_list):
    """Sentence-BERTをFine-Tuningするデータセットを作成する関数"""
    adjacency_pair = list(more_itertools.windowed(spe_label_list, 2, step=1))
    adjacency_pair_df_list = []

    for pair in tqdm(adjacency_pair):
        pair_label_A, pair_label_B = pair

        # speaker label → clause_id を検索して隣接ペアを作る。
        A = _df.filter(pl.col("speaker_label") == pair_label_A)

        B = _df.filter(pl.col("speaker_label") == pair_label_B)
        A_clause_index = A["clause_id"].to_list()
        B_clause_index = B["clause_id"].to_list()

        co_spe_pair_A = list(itertools.combinations(A_clause_index, 2))
        co_spe_pair_B = list(itertools.combinations(B_clause_index, 2))

        clause_all_pair = list(
            itertools.combinations(A_clause_index + B_clause_index, 2)
        )
        co_speaker_pair = list((set(co_spe_pair_A) | set(co_spe_pair_B)))

        clause_pair = list(set(clause_all_pair) - set(co_speaker_pair))

        adjacency_df = pl.DataFrame(
            clause_pair, columns=["anchor", "positive"]
        )
        adjacency_pair_df_list.append(adjacency_df)
        # break

    # co_speaker_pair_df = pl.concat(co_speaker_pair_list)
    adjacency_pair_df = pl.concat(adjacency_pair_df_list)
    adjacency_pair_anchor_id = adjacency_pair_df["anchor"].unique().to_list()

    triple_dataset_list = []

    for anchor_id in tqdm(
        adjacency_pair_anchor_id, total=len(adjacency_pair_anchor_id)
    ):
        positive_ids = adjacency_pair_df.filter(pl.col("anchor") == anchor_id)[
            "positive"
        ].to_list()

        positive_len = len(positive_ids)

        anchor_index = df.filter(pl.col("clause_id") == anchor_id)

        _anchor_df = (
            pl.concat([anchor_index] * positive_len)
            .rename(
                {
                    "clause_id": "anchor_clause_id",
                    "speaker_name": "anchor_speaker_name",
                    "utterance": "anchor_utterance",
                    "speaker_label": "anchor_speaker_label",
                }
            )
            .drop("len")
        )

        positive_index = (
            df.filter(pl.col("clause_id").is_in(positive_ids))
            .rename(
                {
                    "clause_id": "positive_clause_id",
                    "speaker_name": "positive_speaker_name",
                    "utterance": "positive_utterance",
                    "speaker_label": "positive_speaker_label",
                }
            )
            .drop("len")
        )

        other_claims = (
            df.filter(~pl.col("clause_id").is_in(positive_ids + [anchor_id]))
            .sample(n=positive_len)
            .rename(
                {
                    "clause_id": "other_clause_id",
                    "speaker_name": "other_speaker_name",
                    "utterance": "other_utterance",
                    "speaker_label": "other_speaker_label",
                }
            )
            .drop("len")
        )

        triple = pl.concat(
            [_anchor_df, positive_index, other_claims], how="horizontal"
        )
        # negative_sample_list.append(other_claims)
        triple_dataset_list.append(triple)
    return pl.concat(triple_dataset_list)


# %%
data_file_path = "/workspace/data/interim/speaker_utterance_dataset/一般質問(要旨)2月13日/assembly.csv"

# %%
file_path = "/workspace/data/interim/speaker_utterance_dataset"
files = [i for i in os.listdir(file_path) if i not in [".DS_Store"]]

# %%
for file in files:
    df_original = pl.read_csv(f"{file_path}/{file}/assembly.csv").rename(
        {"id": "clause_id", "label": "speaker_label"}
    )

    _df = df_original.filter(
        (pl.col("speaker_name") != "議長") & (pl.col("speaker_name") != "副議長")
    )

    df_add_len = _df.with_columns(
        [pl.col("utterance").apply(len).alias("len")]
    )

    # 10 文字以上で制限？　あまりよくないかも
    index = 10
    df: pl.DataFrame = df_add_len.filter((pl.col("len") > index))

    speaker_label_list = df["speaker_label"].unique().to_list()

    triple_dataset_df = make_triple_pair_process(df, speaker_label_list)
    triple_dataset_df.write_csv(
        f"/workspace/data/interim/adjacency_pair/{file}.csv"
    )

# %%
# adjacency_pair process
triple_dataset_df = make_triple_pair_process(df, speaker_label_list)

# %%
utterance: list = df["utterance"].to_list()
clause_id: list = df["clause_id"].to_list()

clause_id_utterance_dict = dict(zip(clause_id, utterance))


# %%
def search_clauses(_id) -> str:
    x = df.select(["clause_id", "utterance"]).filter(
        pl.col("clause_id") == _id
    )
    return x["utterance"][0]


def clauses_id_to_clauses(_df: pl.DataFrame):
    # _df["A_clause"] = _df["anchor"].apply(lambda x: search_clauses(x))
    dd = _df.with_columns(
        [
            pl.col("anchor")
            .apply(lambda x: clause_id_utterance_dict[x])
            .alias("A_clause"),
            pl.col("positive")
            .apply(lambda x: clause_id_utterance_dict[x])
            .alias("B_clause"),
        ]
    )
    return dd


# %%
adjacency_pair_df_add_clauses = clauses_id_to_clauses(adjacency_pair_df)
adjacency_pair_df_add_clauses.write_csv("adjacency_pair.csv")
co_speaker_pair_df_add_clauses = clauses_id_to_clauses(co_speaker_pair_df)
co_speaker_pair_df_add_clauses.write_csv("co_speaker_pair.csv")
no_pair_df_add_clauses = clauses_id_to_clauses(no_pair_df)
no_pair_df_add_clauses.write_csv("no_pair_df.csv")
