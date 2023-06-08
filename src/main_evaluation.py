# %%
# ライブラリをインポート
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from codebase.dataprocessor.calculate_similarity import (
    chebyshev_distance,
    cos_sim,
    euclidean_distance,
    kl_divergence,
    manhattan_distance,
)
from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_current_datetime
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_csv

# %%
# 環境変数の読み込み
load_dotenv("/workspace/src/.env")
interim_data_path = os.environ["INTERIM_DATA_PATH"]

# %%
# 議事録データフレームを作成
assembly_file_path = "/workspace/data/interim/speaker_utterance_dataset/一般質問(要旨)2月13日/assembly.csv"
df_assembly = data_loder(file_path=assembly_file_path, has_header=True)
utterances = df_assembly["utterance"].to_list()
# probs = np.load("/workspace/src/log/2023-4-13/10-39-20/probs.npy")

# %%
# 議会だよりデータフレームを宣言
digest_file_path: str = "/workspace/data/interim/speaker_utterance_dataset/一般質問(要旨)2月13日/digest.csv"
df_digest = data_loder(file_path=digest_file_path, has_header=True)

# %%
# 話題に対する発話データの作成
# ToDo: 関数にしてcodebaseに移行
df_digest_filtered = df_digest.filter(pl.col("label") != "speaker.summury")
id_num = 0
df_digest_clause = pl.DataFrame(
    {
        "id": [],
        "title": [],
        "speaker_name": [],
        "clause": [],
        "label": [],
    },
    {
        "id": pl.Int64,
        "title": pl.Utf8,
        "speaker_name": pl.Utf8,
        "clause": pl.Utf8,
        "label": pl.Utf8,
    },
)

for i in range(len(df_digest_filtered)):
    if df_digest_filtered["label"][i] == "policy.title":
        title = df_digest_filtered["utterance"][i]

        if title in df_digest_clause["title"].to_list():
            title = title + f"{i}"
    else:
        speaker_name = df_digest_filtered["speaker_name"][i]
        clause = df_digest_filtered["utterance"][i]
        label = df_digest_filtered["label"][i]
        df = pl.DataFrame(
            {
                "id": id_num,
                "title": title,
                "speaker_name": speaker_name,
                "clause": clause,
                "label": label,
            }
        )
        df_digest_clause = pl.concat([df_digest_clause, df])
        id_num += 1

current_date, current_time = get_current_datetime()
save_path = f"{interim_data_path}/digest_clause/{current_date}/{current_time}"
make_dir(path=save_path)
save_file_name = "/evaluate.csv"
save_csv(df=df_digest_clause, path=save_path, file_name=save_file_name)

# %%
# digest_clauseにcos類似度が高い議事録の節を追加
prob_digest_utterance_dir = (
    "/workspace/data/interim/prob_topic_and_on_topic/2023-4-27/20-25-10"
)
prob_clause_dir = "/workspace/data/interim/prob_clause/2023-4-27/7-59-0"
load_path = (
    "/workspace/data/interim/digest_clause/2023-4-20/10-27-4/evaluate.csv"
)
df_digest_clause = data_loder(load_path, has_header=True)
utterances = []

for i in range(len(df_digest_clause)):
    clause = df_digest_clause["clause"][i]
    prob_digest_utterance = np.load(
        f"{prob_digest_utterance_dir}/{clause}/prob.npy"
    )
    results_cos_sim = []

    for utterance in df_assembly["utterance"]:
        prob_clause = np.load(f"{prob_clause_dir}/{utterance}/prob.npy")
        results_cos_sim.append(cos_sim(prob_digest_utterance, prob_clause))

    df = pl.DataFrame({"cos_similarity": results_cos_sim})
    df_assembly_sorted = df_assembly.with_columns(df).sort(
        "cos_similarity", descending=True
    )

    for j in range(len(df_assembly_sorted)):
        if (
            df_digest_clause["speaker_name"][i]
            == df_assembly_sorted["speaker_name"][j]
        ):
            utterance = df_assembly_sorted["utterance"][j]
            break
    utterances.append(utterance)

df = pl.DataFrame({"digest_cos_similarity_assembly_utterance": utterances})
df_digest_clause_add_cos_similarity = df_digest_clause.with_columns(df)

current_date, current_time = get_current_datetime()
save_path = (
    os.environ["INTERIM_DATA_PATH"]
    + f"/digest_clause_add_assembly_utterance/{current_date}/{current_time}"
)
make_dir(path=save_path)
save_file_name = "/evaluate.csv"
save_csv(
    df=df_digest_clause_add_cos_similarity,
    path=save_path,
    file_name=save_file_name,
)

# %%
# だよりに含まれる節の類似度による話題結束性評価用データの作成
# ToDo: 関数にしてcodebaseに移行
load_path = "/workspace/data/interim/digest_clause_add_assembly_utterance/2023-4-28/6-47-36/evaluate.csv"
df_digest_clause = data_loder(load_path, has_header=True)
speaker_name_list = df_digest_clause["speaker_name"].unique().to_list()
num_of_spekaers = len(speaker_name_list)
utterances = df_assembly["utterance"].to_list()

df_policy_and_reply_relation = pl.DataFrame(
    {
        "policy_speaker": [],
        "policy_clause": [],
        "reply_speaker": [],
        "reply_clause": [],
        "label": [],
    },
    {
        "policy_speaker": pl.Utf8,
        "policy_clause": pl.Utf8,
        "reply_speaker": pl.Utf8,
        "reply_clause": pl.Utf8,
        "label": pl.Int64,
    },
)

for i in range(0, num_of_spekaers - 1):
    df_speaker_name_filtered_one = df_digest_clause.filter(
        pl.col("speaker_name") == speaker_name_list[i]
    )
    for j in range(i + 1, num_of_spekaers):
        df_speaker_name_filtered_two = df_digest_clause.filter(
            pl.col("speaker_name") == speaker_name_list[j]
        )
        for k in range(len(df_speaker_name_filtered_one)):
            for l in range(len(df_speaker_name_filtered_two)):
                label = 0
                if (
                    df_speaker_name_filtered_one["title"][k]
                    == df_speaker_name_filtered_two["title"][l]
                ):
                    label = 1
                df = pl.DataFrame(
                    {
                        "policy_speaker": speaker_name_list[i],
                        "policy_clause": df_speaker_name_filtered_one[
                            # "digest_cos_similarity_assembly_utterance_id"
                            "clause"
                        ][k],
                        "reply_speaker": speaker_name_list[j],
                        "reply_clause": df_speaker_name_filtered_two[
                            # "digest_cos_similarity_assembly_utterance_id"
                            "clause"
                        ][l],
                        "label": label,
                    }
                )

                df_policy_and_reply_relation = pl.concat(
                    [df_policy_and_reply_relation, df]
                )

current_date, current_time = get_current_datetime()
save_path = (
    os.environ["INTERIM_DATA_PATH"]
    + f"/policy_and_reply_clause_relation/{current_date}/{current_time}"
)
make_dir(path=save_path)
save_file_name = "/evaluate.csv"
save_csv(
    df=df_policy_and_reply_relation, path=save_path, file_name=save_file_name
)

# %%
# だよりに含まれる節の類似度による話題結束性評価
# ToDo: 関数にしてcodebaseに移行
load_path = "/workspace/data/interim/policy_and_reply_clause_relation/2023-4-28/11-27-1/evaluate.csv"
df_policy_and_reply_relation = data_loder(load_path, has_header=True)
prob_clause_dir = "/workspace/data/interim/prob_clause/2023-4-27/7-59-0"
prob_digest_dir = (
    "/workspace/data/interim/prob_topic_and_on_topic/2023-4-27/20-25-10"
)
cos_similaritys_list = []
n = 3

for i in range(len(df_policy_and_reply_relation)):
    # if i > 2:
    #     break
    policy_clause = df_policy_and_reply_relation["policy_clause"][i]
    reply_clause = df_policy_and_reply_relation["reply_clause"][i]
    prob_policy_clause = np.load(f"{prob_digest_dir}/{policy_clause}/prob.npy")
    prob_reply_clause = np.load(f"{prob_digest_dir}/{reply_clause}/prob.npy")

    df_assembly_filtered_policy_speaker = df_assembly.filter(
        pl.col("speaker_name")
        == df_policy_and_reply_relation["policy_speaker"][i]
    )
    df_assembly_filtered_reply_speaker = df_assembly.filter(
        pl.col("speaker_name")
        == df_policy_and_reply_relation["reply_speaker"][i]
    )

    prob_policy_utterance_list = []
    for policy_utterance in df_assembly_filtered_policy_speaker["utterance"]:
        prob_clause = np.load(f"{prob_clause_dir}/{policy_utterance}/prob.npy")
        prob_policy_utterance_list.append(
            cos_sim(prob_policy_clause, prob_clause)
        )
    df_cos_sim = pl.DataFrame({"cos_similarity": prob_policy_utterance_list})
    df_policy_utterance_sorted = (
        df_assembly_filtered_policy_speaker.with_columns(df_cos_sim)
        .sort("cos_similarity", descending=True)
        .head(n)
    )
    prob_policy_utterance_ave = np.zeros(len(prob_policy_clause))
    for j, policy_utterance in enumerate(
        df_policy_utterance_sorted["utterance"]
    ):
        prob_policy_utterance_ave += np.load(
            f"{prob_clause_dir}/{policy_utterance}/prob.npy"
        )
    prob_policy_utterance_ave /= n

    prob_reply_utterance_list = []
    for reply_utterance in df_assembly_filtered_reply_speaker["utterance"]:
        prob_clause = np.load(f"{prob_clause_dir}/{reply_utterance}/prob.npy")
        prob_reply_utterance_list.append(
            cos_sim(prob_reply_clause, prob_clause)
        )
    df_cos_sim = pl.DataFrame({"cos_similarity": prob_reply_utterance_list})
    df_reply_utterance_sorted = (
        df_assembly_filtered_reply_speaker.with_columns(df_cos_sim)
        .sort("cos_similarity", descending=True)
        .head(n)
    )
    prob_reply_utterance_ave = np.zeros(len(prob_reply_clause))
    for k, reply_utterance in enumerate(
        df_reply_utterance_sorted["utterance"]
    ):
        prob_reply_utterance_ave += np.load(
            f"{prob_clause_dir}/{reply_utterance}/prob.npy"
        )
    prob_reply_utterance_ave /= n

    cos_similaritys_list.append(cos_sim(prob_policy_clause, prob_reply_clause))

df_prob = pl.DataFrame(
    {
        "cos_similarity": cos_similaritys_list,
    }
)
df_policy_and_reply_relation = df_policy_and_reply_relation.with_columns(
    df_prob
)

current_date, current_time = get_current_datetime()
save_path = (
    os.environ["INTERIM_DATA_PATH"]
    + f"/prob_policy_and_reply_clause_relation/{current_date}/{current_time}"
)
make_dir(path=save_path)
save_file_name = "/evaluate.csv"
save_csv(
    df=df_policy_and_reply_relation, path=save_path, file_name=save_file_name
)

# %%
# ROC曲線の計算
# 関数にしてcodebaseに移行
load_path = "/workspace/data/interim/prob_policy_and_reply_clause_relation/2023-4-29/5-52-47/evaluate.csv"
df_policy_and_reply_relation = data_loder(load_path, has_header=True)
fpr, tpr, thresholds = roc_curve(
    df_policy_and_reply_relation["label"].to_list(),
    df_policy_and_reply_relation["cos_similarity"].to_list(),
)

plt.plot(fpr, tpr)
plt.xlabel("FPR: False positive rate")
plt.ylabel("TPR: True positive rate")
plt.grid()

print(
    roc_auc_score(
        df_policy_and_reply_relation["label"].to_list(),
        df_policy_and_reply_relation["cos_similarity"].to_list(),
    )
)

# %%
# 会議録の節と都議会だよりの話題の話者一致評価
# ToDo: 関数にしてcodebaseに移行
cos_sim_match_count = 0
kl_dive_match_count = 0
full_count = 0
df_digest_filtered = df_digest.filter(pl.col("label") != "speaker.summury")
probs = np.load("/workspace/src/log/2023-4-13/10-39-20/probs.npy")

for i in range(len(df_digest_filtered)):
    if df_digest_filtered["label"][i] == "policy.title":
        query_prob = np.load(
            "/workspace/data/interim/prob_title/2023-4-18/10-45-47/"
            + df_digest_filtered["utterance"][i]
            + "/prob.npy"
        )
        cos_sims = [cos_sim(prob, query_prob) for prob in probs]
        kl_dive = [kl_divergence(prob, query_prob) for prob in probs]
        df_cos_sim = pl.DataFrame({"cos_similarity": cos_sims})
        df_kl_dive = pl.DataFrame({"kl_divergence": kl_dive})
        df_add_cos_sim = df_assembly.with_columns(df_cos_sim)
        df_add_kl_dive = df_assembly.with_columns(df_kl_dive)
        df_sort_cos_sim = df_add_cos_sim.sort(
            "cos_similarity", descending=True
        ).head(5)
        df_sort_kl_dive = df_add_kl_dive.sort(
            "kl_divergence", descending=True
        ).tail(5)
    else:
        full_count += 1
        if (
            df_digest_filtered["speaker_name"][i]
            in df_sort_cos_sim["speaker_name"].to_list()
        ):
            cos_sim_match_count += 1
        if (
            df_digest_filtered["speaker_name"][i]
            in df_sort_kl_dive["speaker_name"].to_list()
        ):
            kl_dive_match_count += 1

print(f"cos similarity match rate: {cos_sim_match_count / full_count}")
print(f"kl divergence match rate: {kl_dive_match_count / full_count}")

# %%
# 都議会だよりの話題と話題に対する節の関係評価用データフレームの作成
# ToDo: 関数にしてcodebaseに移行
load_dotenv("/workspace/src/.env")
digest_file_path: str = "/workspace/data/interim/speaker_utterance_dataset/一般質問(要旨)2月13日/digest.csv"
df_digest = data_loder(file_path=digest_file_path, has_header=True)
df_digest_filtered = df_digest.filter(pl.col("label") != "speaker.summury")
df_digest_titles = df_digest_filtered.filter(pl.col("label") == "policy.title")
df_digest_clause = df_digest_filtered.filter(pl.col("label") != "policy.title")
df_full = pl.DataFrame(
    {
        "id": [],
        "title": [],
        "clause": [],
        "cos_similarity": [],
        "kl_divergence": [],
        "chebyshev_distance": [],
        "euclidean_distance": [],
        "manhattan_distance": [],
    },
    {
        "id": pl.Int64,
        "title": pl.Utf8,
        "clause": pl.Utf8,
        "cos_similarity": pl.Float64,
        "kl_divergence": pl.Float64,
        "chebyshev_distance": pl.Float64,
        "euclidean_distance": pl.Float64,
        "manhattan_distance": pl.Float64,
    },
)
id_num = 0

for title in tqdm(df_digest_titles["utterance"]):
    prob_title = np.load(
        "/workspace/data/interim/prob_title/2023-4-18/10-45-47/"
        + title
        + "/prob.npy"
    )
    for clause in df_digest_clause["utterance"]:
        prob_clause = np.load(
            "/workspace/data/interim/prob_topic_and_on_topic/2023-4-16-8-45-1/"
            + clause
            + "/prob.npy"
        )
        cos_sims = cos_sim(prob_title, prob_clause)
        kl_dive = kl_divergence(prob_title, prob_clause)
        chebyshev_dist = chebyshev_distance(prob_title, prob_clause)
        euclidean_dist = euclidean_distance(prob_title, prob_clause)
        manhattan_dist = manhattan_distance(prob_title, prob_clause)
        df = pl.DataFrame(
            {
                "id": id_num,
                "title": title,
                "clause": clause,
                "cos_similarity": cos_sims,
                "kl_divergence": kl_dive,
                "chebyshev_distance": chebyshev_dist,
                "euclidean_distance": euclidean_dist,
                "manhattan_distance": manhattan_dist,
            }
        )
        df_full = pl.concat([df_full, df])
        id_num += 1

path = os.environ["LOG_PATH"]
file_name = "test.csv"
save_csv(df_full, path, file_name)

# %%
# 都議会だよりの話題と話題に対する節の関係評価
# ToDo: 関数にしてcodebaseに移行
topic_and_clause_on_topic_file_path: str = "/workspace/src/log/test.csv"
df_topic_and_clause_on_topic = data_loder(
    file_path=topic_and_clause_on_topic_file_path, has_header=True
)
flag = 0
n = 5
full_count = 0
cos_sim_match_count = 0
kl_dive_match_count = 0
chebyshev_dist_match_count = 0
euclidean_dist_match_count = 0
manhattan_dist_match_count = 0

for i, label in enumerate(df_digest_filtered["label"]):
    if label == "policy.title":
        df_topic_and_clause_on_topic_filtered = (
            df_topic_and_clause_on_topic.filter(
                pl.col("title") == df_digest_filtered["utterance"][i]
            )
        )
        if len(df_topic_and_clause_on_topic_filtered) > 114 and flag == 0:
            df_topic_and_clause_on_topic_filtered = (
                df_topic_and_clause_on_topic_filtered[:114]
            )
            flag = 1
        elif len(df_topic_and_clause_on_topic_filtered) > 114 and flag == 1:
            df_topic_and_clause_on_topic_filtered = (
                df_topic_and_clause_on_topic_filtered[114:]
            )

        df_topic_and_clause_on_topic_sort_cos_sim = (
            df_topic_and_clause_on_topic_filtered.sort(
                "cos_similarity", descending=True
            ).head(n)
        )
        df_topic_and_clause_on_topic_sort_kl_dive = (
            df_topic_and_clause_on_topic_filtered.filter(
                pl.col("kl_divergence") != 0.0
            )
            .sort("kl_divergence")
            .head(n)
        )
        df_topic_and_clause_on_topic_sort_chebyshev_dist = (
            df_topic_and_clause_on_topic_filtered.filter(
                pl.col("chebyshev_distance") != 0.0
            )
            .sort("chebyshev_distance")
            .head(n)
        )
        df_topic_and_clause_on_topic_sort_euclidean_dist = (
            df_topic_and_clause_on_topic_filtered.filter(
                pl.col("euclidean_distance") != 0.0
            )
            .sort("euclidean_distance")
            .head(n)
        )
        df_topic_and_clause_on_topic_sort_manhattan_dist = (
            df_topic_and_clause_on_topic_filtered.filter(
                pl.col("manhattan_distance") != 0.0
            )
            .sort("manhattan_distance")
            .head(n)
        )
    else:
        full_count += 1
        if (
            df_digest_filtered["utterance"][i]
            in df_topic_and_clause_on_topic_sort_cos_sim["clause"].to_list()
        ):
            cos_sim_match_count += 1
            # print(df_digest_filtered["utterance"][i])
            # print(df_topic_and_clause_on_topic_sort["clause"].to_list())
        if (
            df_digest_filtered["utterance"][i]
            in df_topic_and_clause_on_topic_sort_kl_dive["clause"].to_list()
        ):
            kl_dive_match_count += 1
            # print(df_digest_filtered["utterance"][i])
            # print(
            #     df_topic_and_clause_on_topic_sort_kl_dive["clause"].to_list()
            # )
        if (
            df_digest_filtered["utterance"][i]
            in df_topic_and_clause_on_topic_sort_chebyshev_dist[
                "clause"
            ].to_list()
        ):
            chebyshev_dist_match_count += 1
            # print(df_digest_filtered["utterance"][i])
            # print(
            #     df_topic_and_clause_on_topic_sort_chebyshev_dist["clause"].to_list()
            # )
        if (
            df_digest_filtered["utterance"][i]
            in df_topic_and_clause_on_topic_sort_euclidean_dist[
                "clause"
            ].to_list()
        ):
            euclidean_dist_match_count += 1
            # print(df_digest_filtered["utterance"][i])
            # print(
            #     df_topic_and_clause_on_topic_sort_euclidean_dist["clause"].to_list()
            # )
        if (
            df_digest_filtered["utterance"][i]
            in df_topic_and_clause_on_topic_sort_manhattan_dist[
                "clause"
            ].to_list()
        ):
            manhattan_dist_match_count += 1
            # print(df_digest_filtered["utterance"][i])
            # print(
            #     df_topic_and_clause_on_topic_sort_manhattan_dist["clause"].to_list()
            # )

print(f"full_count: {full_count}")
print(f"cos similarity match_count: {cos_sim_match_count}")
print(f"cos similarity match rate: {cos_sim_match_count / full_count}\n")

print(f"full_count: {full_count}")
print(f"kl divergence match_count: {kl_dive_match_count}")
print(f"kl divergence match rate: {kl_dive_match_count / full_count}\n")

print(f"full_count: {full_count}")
print(f"chebyshev distance match_count: {chebyshev_dist_match_count}")
print(
    f"chebyshev distance match rate: {chebyshev_dist_match_count / full_count}\n"
)

print(f"full_count: {full_count}")
print(f"euclidean distance match_count: {euclidean_dist_match_count}")
print(
    f"euclidean distance match rate: {euclidean_dist_match_count / full_count}\n"
)

print(f"full_count: {full_count}")
print(f"manhattan distance match_count: {manhattan_dist_match_count}")
print(
    f"manhattan distance match rate: {manhattan_dist_match_count / full_count}\n"
)

# %%
