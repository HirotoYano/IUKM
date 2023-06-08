import polars as pl
from sklearn import preprocessing


def label_encoder(df):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(df["column_1"])

    return df.with_columns(pl.Series(labels).alias("column_4"))
