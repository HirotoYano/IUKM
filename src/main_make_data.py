# %%
# ライブラリのインポート
import os
import re

from dotenv import load_dotenv
from tqdm import tqdm

from codebase.dataprocessor.data_loder import data_loder
from codebase.dataprocessor.get_info import get_file_name
from codebase.dataprocessor.make_dir import make_dir
from codebase.dataprocessor.make_speaker_utterance_data import make_assembly_data, make_digest_data
from codebase.dataprocessor.save_data import save_csv

# %%
# 環境変数の読み込み
load_dotenv("/workspace/src/.env")
raw_data_path = os.environ["RAW_DATA_PATH"]
minutes_json_data = os.environ["MINUTES_JSON_DATA"]
interim_data_path = os.environ["INTERIM_DATA_PATH"]

# %%
# ファイル名を取得
get_file_name_dir_path = raw_data_path + "/" + minutes_json_data
files = get_file_name(os.listdir(get_file_name_dir_path))

# %%
# 必要なファイル名を抽出
files = [file for file in files if re.match("一般質問", file)]

# %%
# 隣接ペアデータセットを作成


# %%
# CSVファイルを保存
for file in tqdm(files):
    load_data_path = raw_data_path + "/" + minutes_json_data + "/" + file
    minutes_json = data_loder(file_path=load_data_path)
    df_assembly = make_assembly_data(minutes_json)
    df_digest = make_digest_data(minutes_json)

    make_dir_path = interim_data_path + "/speaker_utterance_dataset/" + os.path.splitext(file)[0]
    make_dir(make_dir_path)

    save_csv(df=df_assembly, path=make_dir_path, file_name="/assembly.csv")
    save_csv(df=df_digest, path=make_dir_path, file_name="/digest.csv")

# %%
