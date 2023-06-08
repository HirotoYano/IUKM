# %%
# ライブラリをインポート
from __future__ import annotations

import os

from dotenv import load_dotenv

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.data_split import data_split
from codebase.dataprocessor.label_encoder import label_encoder
from codebase.model.sentence_bert import intra_speaker_model
from codebase.trainer.model_fit import sentence_bert_fit

# %%
# 環境変数の読み込み
load_dotenv("/workspace/src/.env")

# %%
# JSNLIデータの読み込み
data_path = os.environ["RAW_DATA_PATH"] + "/" + os.environ["JSNLI_DATA"]
df = data_loder(data_path)

# %%
# ラベル変換
df_add_label = label_encoder(df)

# %%
# データ分割
df_train, df_eval, df_test = data_split(df_add_label)

# %%
# intra speaker
model = intra_speaker_model(os.environ["PRE_TRAINED_MODEL"])

# %%
# トレーニング
sentence_bert_fit(df_train=df_train, model=model)

# %%
