""" data split function """
import polars as pl


def data_split(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_size: int = int(len(df) * 0.8)
    eval_size: int = train_size + int(len(df) * 0.1)
    return df[0:train_size], df[train_size:eval_size], df[eval_size:]
