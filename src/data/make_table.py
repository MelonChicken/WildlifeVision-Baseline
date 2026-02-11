import pandas as pd
import os
from src.config.paths import *


def make_train_table(
        autosave=True,
) -> pd.DataFrame:
    """
    make single train table
    :param autosave:
    :return: pd.DataFrame
    """

    df_train_features = pd.read_csv(os.path.join(DATA_DIR,'train_features.csv'), index_col=0)
    df_train_labels = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'), index_col=0)

    # check data tables
    # print(f"number of training samples: {len(df_train_features)}")
    # print(df_train_features.head())

    # print(f"number of training labels: {len(df_train_labels)}")
    # print(df_train_labels.head())

    # merged train table
    df_merged = pd.merge(df_train_features, df_train_labels, on='id')

    # print(f"number of lines in training table: {len(df_merged)}")
    df_merged['filepath'] = 'data/' + df_merged['filepath'].astype(str)

    # print(df_merged.head())
    if autosave:
        df_merged.to_csv(os.path.join(ARTIFACTS_DIR, 'processed', 'train_table.csv'))

    if not(len(df_merged) == len(df_train_features) & len(df_merged) == len(df_train_labels)):
        raise Exception("number of lines in training table and training features do not match")

    return df_merged
