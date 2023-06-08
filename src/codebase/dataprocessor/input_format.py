"""this source is sentence input format for sentence bert model."""
from typing import List


class Input_Format:
    """
    Sentence-BERTに入力するデータ形式を指定
    """

    def __init__(
        self, guid: str = "0", texts: List[str] = ["None"], label: int = 0
    ):
        """
        # guid
            番号
        # texts
            テキスト
        # label
            ラベル
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> guid: {}, label: {}, texts: {}".format(
            str(self.guid), str(self.label), "; ".join(self.texts)
        )
