import numpy as np


def save_csv(df, path, file_name):
    df.write_csv(f"{path}/{file_name}")


def save_prob(prob, save_dir_path, save_file_name):
    np.save(f"{save_dir_path}/{save_file_name}", prob)
