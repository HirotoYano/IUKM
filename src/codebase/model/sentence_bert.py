# %%
import math
from datetime import datetime
from typing import List

import transformers
from sentence_transformers import (
    SentencesDataset,
    SentenceTransformer,
    losses,
    models,
)
from sentence_transformers.evaluation import (
    LabelAccuracyEvaluator,
    ParaphraseMiningEvaluator,
    TripletEvaluator,
)

# from torch import nn
from torch.utils.data import DataLoader

transformers.BertTokenizer = transformers.BertJapaneseTokenizer


# %%
class Sentence_BERT_Model(object):
    """setntece bert model class"""

    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        dense_dim: int = 5,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        use_dirver: str = "cuda:0",
        model_save_file_path: str = "./output",
    ):
        self.model_name: str = model_name
        self.max_seq_length: int = max_seq_length
        self.pooling_mode_mean_tokens: bool = pooling_mode_mean_tokens
        self.pooling_mode_cls_token: bool = pooling_mode_cls_token
        self.pooling_mode_max_tokens: bool = pooling_mode_max_tokens
        self.use_dirver: str = use_dirver
        self.model_save_file_path: str = model_save_file_path
        self.dense_dim: int = dense_dim
        self.model: SentenceTransformer = self.make_model(self.dense_dim)

    def make_model(self, n_dim: int) -> SentenceTransformer:
        word_embedding_model = models.Transformer(
            self.model_name, max_seq_length=self.max_seq_length
        )

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=self.pooling_mode_mean_tokens,
            pooling_mode_cls_token=self.pooling_mode_cls_token,
            pooling_mode_max_tokens=self.pooling_mode_max_tokens,
        )

        # dense = models.Dense(
        #     in_features=word_embedding_model.get_word_embedding_dimension(),
        #     out_features=n_dim,
        #     activation_function=nn.Tanh(),
        # )

        model = SentenceTransformer(
            modules=[
                word_embedding_model,
                pooling_model,
                # dense,
            ],
            device=self.use_dirver,
        )

        return model

    def dataloader(self, data, batch_size):
        return DataLoader(
            data, shuffle=True, batch_size=batch_size, num_workers=10
        )

    def prediction(self):
        pass

    def load_data_for_paraphrase_mining(self, dataset):
        sentences_map = {}  # id -> sent
        sentences_reverse_map = {}  # sent -> id
        duplicates_list = []  # (id1, id2)

        def register(sent):
            if sent not in sentences_reverse_map:
                _id = str(len(sentences_reverse_map))
                sentences_reverse_map[sent] = _id
                sentences_map[_id] = sent
                return _id
            else:
                return sentences_reverse_map[sent]

        for data in dataset:
            texts: list = data.texts
            ids = [register(sent) for sent in texts]
            duplicates_list.append(tuple(ids[:2]))

        return sentences_map, duplicates_list

    def triple_evaluator_dataset(self, dataset):
        triple_dataset_T = [
            [i.texts[0], i.texts[1], i.texts[2]] for i in dataset
        ]
        anchor, positive, negative = [list(i) for i in zip(*triple_dataset_T)]
        return anchor, positive, negative

    def train(
        self,
        dataset: List,
        label_num=1,
        batch_size: int = 256,
        num_epochs: int = 10,
        evaluation_steps=1000,
        loss_finction=None,
        evaluat_function=None,
        dataset_name: str = "",
    ):
        date_time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        # model_save_path = f"sbert_output/{self.model_name}/dim_{self.dense_dim}/{dataset_name}"

        sentence_dataset = SentencesDataset(dataset, model=self.model)

        train_data = test_data = vali_data = sentence_dataset

        train_dataloader = self.dataloader(train_data, batch_size)
        # test_dataloader = self.dataloader(test_data, batch_size)
        vali_dataloader = self.dataloader(vali_data, batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        # loss
        if (loss_finction == None) or loss_finction == "softmax":
            train_loss = losses.SoftmaxLoss(
                model=self.model,
                sentence_embedding_dimension=self.dense_dim,
                num_labels=label_num,
            )
        elif loss_finction == "MultipleNegativesRankingLoss":
            train_loss = losses.MultipleNegativesRankingLoss(self.model)
        else:
            print("no loss finction")
            assert 1 == 3

        # evaluator
        evaluator = LabelAccuracyEvaluator(
            vali_dataloader,
            softmax_model=train_loss,
            name="val",
        )
        if (evaluat_function == None) or (
            evaluat_function == "LabelAccuracyEvaluator"
        ):
            evaluator = LabelAccuracyEvaluator(
                vali_dataloader,
                softmax_model=train_loss,
                name="val",
            )
        elif evaluat_function == "ParaphraseMiningEvaluator":
            (
                sentences_map,
                duplicates_list,
            ) = self.load_data_for_paraphrase_mining(dataset)
            evaluator = ParaphraseMiningEvaluator(
                sentences_map, duplicates_list, name="paramin-jsnli-dev"
            )
        elif evaluat_function == "TripletEvaluator":
            anchor, positive, negative = self.triple_evaluator_dataset(dataset)
            evaluator = TripletEvaluator(
                anchors=anchor,
                positives=positive,
                negatives=negative,
                name=f"{evaluat_function}",
                show_progress_bar=True,
                batch_size=batch_size,
            )
        else:
            assert 1 == 4

        # 10% of train data
        warmup_steps = math.ceil(len(train_dataloader) * 0.01)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=self.model_save_file_path
            + "/log/sentence_bert/"
            + date_time,
            checkpoint_path=self.model_save_file_path + "/checkpoints",
            checkpoint_save_steps=evaluation_steps,
        )

    def get_model(self):
        return self.model

    def show_settings(self):
        print(vars(self))


# sentence BERTの定義
def intra_speaker_model(pre_trained_model):
    word_embedding_model = models.Transformer(
        pre_trained_model, max_seq_length=256
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def inter_speaker_model(pre_trained_model):
    word_embedding_model = models.Transformer(
        pre_trained_model, max_seq_length=256
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def model_load(model_path):
    return SentenceTransformer(model_path)
