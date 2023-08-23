""" dataloder function """
import json
import os
import re
from typing import List

import polars as pl

from codebase.dataprocessor.input_format import Input_Format


def data_loder(file_path: str, has_header: bool = False) -> pl.DataFrame:
    if re.compile(os.path.splitext(file_path)[-1]).search(".tsv"):
        return pl.read_csv(file_path, separator="\t", has_header=has_header)
    elif re.compile(os.path.splitext(file_path)[-1]).search(".csv"):
        return pl.read_csv(file_path, has_header=has_header)
    elif re.compile(os.path.splitext(file_path)[-1]).search(".json"):
        json_open = open(file_path, "r")
        return json.load(json_open)
    else:
        print(f"{file_path}は入力可能な形式ではありません。\n入力可能な形式のファイル名を入力してください。")


class Assembly_Triple_Dataloader(object):
    """都議会会議録の発言を隣接ペアとその他の３つの節を読み込む関数"""

    def __init__(self, folder_name: str) -> None:
        self.folder_name: str = folder_name

    def get_examples(self, file_name: str, max_examples=0) -> List[Input_Format]:
        read_df = pl.read_csv(self.folder_name + "/" + file_name)
        sentence1 = read_df["anchor_utterance"].to_list()
        sentence2 = read_df["positive_utterance"].to_list()
        sentence3 = read_df["other_utterance"].to_list()

        examples = []
        _id = 0
        for anchor, positive, negative in zip(sentence1, sentence2, sentence3):
            # print(s1, s2, label)
            anc = str(anchor)
            pos = str(positive)
            neg = str(negative)
            guid = "%s-%d" % (file_name, _id)
            _id += 1
            examples.append(Input_Format(guid=guid, texts=[anc, pos, neg]))
            if 0 < max_examples <= len(examples):
                break

        return examples
