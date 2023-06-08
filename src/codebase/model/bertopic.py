from bertopic import BERTopic


def bertopic_model():
    return BERTopic(
        language="japanese",
        calculate_probabilities=True,
        verbose=True,
        # nr_topics="49",
    )


def save_model(model, save_dir_path, save_file_name):
    model.save(f"{save_dir_path}/{save_file_name}")


def load_model(model_path):
    return BERTopic.load(model_path)
