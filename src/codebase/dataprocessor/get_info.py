import datetime

from codebase.model.bertopic import load_model


def get_file_name(target_dir):
    return [file for file in target_dir]


def get_current_datetime():
    dt_now = datetime.datetime.now()
    current_date = f"{dt_now.year}-{dt_now.month}-{dt_now.day}"
    current_time = f"{dt_now.hour}-{dt_now.minute}-{dt_now.second}"

    return current_date, current_time


def get_probability_distribution(model_path, doc):
    topic_model = load_model(model_path=model_path)
    return topic_model.transform(doc)
