from __future__ import annotations

import datetime
import os

from dotenv import load_dotenv
from sentence_transformers import InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader


def sentence_bert_fit(df_train, model):
    load_dotenv("/workspace/src/.env")
    dt_now = datetime.datetime.now()
    CHECKPOINT_PATH: str = f"/workspace/src/checkpoints/{dt_now.year}-{dt_now.month}-{dt_now.day}/{dt_now.hour}-{dt_now.minute}-{dt_now.second}"
    OUTPUT_PATH: str = f"/workspace/src/log/{dt_now.year}-{dt_now.month}-{dt_now.day}/{dt_now.hour}-{dt_now.minute}-{dt_now.second}"
    WARMUP_STEPS: int = int(
        len(df_train) // int(os.environ["BATCH_SIZE"]) * 0.1
    )

    train_examples = [
        InputExample(
            texts=[column_2, column_3],
            label=column_4,
        )
        for _, column_2, column_3, column_4 in df_train.iterrows()
    ]

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=int(os.environ["BATCH_SIZE"])
    )
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=int(os.environ["TRAIN_NUM_LABELS"]),
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=int(os.environ["NUM_EPOCHS"]),
        warmup_steps=WARMUP_STEPS,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=OUTPUT_PATH,
    )


def bertopic_fit_transform(model, docs):
    return model.fit_transform(docs)
