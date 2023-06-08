# %%
# ライブラリをインポート
import os

import polars as pl
from dotenv import load_dotenv

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import (
    get_current_datetime,
    get_probability_distribution,
)
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.save_data import save_prob
from codebase.model.bertopic import bertopic_model, save_model
from codebase.trainer.model_fit import bertopic_fit_transform

# %%
# 学習前環境変数の読み込み
load_dotenv("/workspace/src/.env")
log_path = os.environ["LOG_PATH"]
interim_data_path = os.environ["INTERIM_DATA_PATH"]

# %%
# BERTopicを定義
topic_model = bertopic_model()

# %%
# 節リストを作成
assembly_file_path = "/workspace/data/interim/speaker_utterance_dataset/一般質問(要旨)2月13日/assembly.csv"
df_assembly = data_loder(file_path=assembly_file_path, has_header=True)
utterances = df_assembly["utterance"].to_list()

# %%
# fit
topics, probs = bertopic_fit_transform(model=topic_model, docs=utterances)
current_date, current_time = get_current_datetime()

save_dir_path = f"{log_path}/{current_date}/{current_time}"
make_dir(save_dir_path)
save_model(
    model=topic_model, save_dir_path=save_dir_path, save_file_name="model.bin"
)

# %%
# 学習後の環境変数の読み込み
load_dotenv("/workspace/src/.env")
bertopic_model_path = (
    os.environ["LOG_PATH"] + os.environ["BERTOPIC_MODEL_PATH"]
)
interim_data_path = os.environ["INTERIM_DATA_PATH"]

# %%
# 議事録の節の確率分布を保存
current_date, current_time = get_current_datetime()
assembly_file_path = "/workspace/data/interim/speaker_utterance_dataset/一般質問(要旨)2月13日/assembly.csv"
df_assembly = data_loder(file_path=assembly_file_path, has_header=True)
utterances = df_assembly["utterance"].to_list()
save_dir = "/workspace/data/interim/prob_clause/2023-4-27/7-59-0"
list_dirs = os.listdir(save_dir)

for i, utterance in enumerate(utterances):
    # if i > 1:
    #     break
    if utterance not in list_dirs:
        print(f"{i}. add utterance: {utterance}")
        prob = get_probability_distribution(
            model_path=bertopic_model_path, doc=utterance
        )
        # save_dir_path = f"{interim_data_path}/prob_clause/{current_date}/{current_time}/{utterance}"
        save_dir_path = f"{save_dir}/{utterance}"
        make_dir(path=save_dir_path)
        save_prob(
            prob=prob[1][0], save_dir_path=save_dir_path, save_file_name="prob"
        )

# %%
# 議会だよりの話題と話題に対する節の確率分布をそれぞれ保存
current_date, current_time = get_current_datetime()
digest_file_path: str = "/workspace/data/interim/speaker_utterance_dataset/一般質問(要旨)2月13日/digest.csv"
df_digest = data_loder(file_path=digest_file_path, has_header=True)
df_digest_filtered = df_digest.filter(pl.col("label") != "speaker.summury")

for i, utterance in enumerate(df_digest_filtered["utterance"]):
    # if i > 1:
    #     break
    prob = get_probability_distribution(
        model_path=bertopic_model_path, doc=utterance
    )

    if df_digest_filtered["label"][i] == "policy.title":
        save_dir_path = f"{interim_data_path}/prob_title/{current_date}/{current_time}/{utterance}"
        make_dir(path=save_dir_path)
        save_prob(
            prob=prob[1][0],
            save_dir_path=save_dir_path,
            save_file_name="prob",
        )
    else:
        save_dir_path = f"{interim_data_path}/prob_topic_and_on_topic/{current_date}/{current_time}/{utterance}"
        make_dir(path=save_dir_path)
        save_prob(
            prob=prob[1][0],
            save_dir_path=save_dir_path,
            save_file_name="prob",
        )

# %%
