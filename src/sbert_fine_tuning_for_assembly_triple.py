# # %%
import hydra

# import os
from omegaconf import DictConfig

from codebase.dataprocessor.data_loder import Assembly_Triple_Dataloader
from codebase.model.sentence_bert import Sentence_BERT_Model

# from hydra.utils import to_absolute_path, get_original_cwd


# # %%
def main(
    dataset_file_path,
    dataset_file_name,
    dense_dim,
    model_name="cl-tohoku/bert-base-japanese-whole-word-masking",
    max_seq_length: int = 256,
    batch_size: int = 256,
    use_dirver: str = "cuda:0",
    num_epochs: int = 10,
    evaluation_steps: int = 500,
    loss_finction: str = "MultipleNegativesRankingLoss",
    evaluat_function: str = "TripletEvaluator",
    output_path: str = "/workspace/src",
):
    dataloader = Assembly_Triple_Dataloader(dataset_file_path)
    dataset = dataloader.get_examples(dataset_file_name)

    sbert_model = Sentence_BERT_Model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        use_dirver=use_dirver,
        dense_dim=dense_dim,
        model_save_file_path=output_path,
    )

    sbert_model.show_settings()

    sbert_model.train(
        dataset=dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        evaluation_steps=evaluation_steps,
        loss_finction=loss_finction,
        evaluat_function=evaluat_function,
        dataset_name=dataset_file_name,
    )
    del sbert_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig):
    main(
        dataset_file_path=cfg.model.dataset_file_path,
        dataset_file_name=cfg.model.dataset_file_name,
        dense_dim=cfg.model.dense_dim,
        model_name=cfg.model.model_name,
        max_seq_length=cfg.model.max_seq_length,
        use_dirver=cfg.model.use_dirver,
        num_epochs=cfg.model.num_epochs,
        evaluation_steps=cfg.model.evaluation_steps,
        loss_finction=cfg.model.loss_finction,
        evaluat_function=cfg.model.evaluat_function,
    )


if __name__ == "__main__":
    my_app()
