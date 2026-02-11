import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config.paths import ARTIFACTS_DIR


def add_group_folds(
    df: pd.DataFrame,
    *,
    group_col: str = "site",
    n_splits: int = 5,
    fold_col: str = "fold",
    shuffle: bool = False,
    autosave: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    add group fold column to the train table to ensure the situation that the same site(group) appears in both train and valid data

    :param fold_col:
    :param df: pd.DataFrame
    :param group_col: str
    :param n_splits: int : default n_splits is 5 which usually results in the normal performance in training process
    :param shuffle: bool
    :param autosave:
    :param random_state: int

    :return: pd.DataFrame
    """

    df_output = df.copy()
    # reset the row index to prevent indexing collision issue
    df_output = df_output.reset_index(drop=True)
    df_output[fold_col] = -1 # initialize the value

    # X and y are not mandatory to split, just for indexing data
    # dummy feature to satisfy the number of samples(len(out)) and its initial value as 1
    X = np.zeros((len(df_output), 1))
    # dummy feature to satisfy the number of samples
    y = np.zeros(len(df_output))

    group_k_fold_splitter = GroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state)

    groups = df_output[group_col].values

    # splitter will automatically gather sites which has the similar number of samples
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
    for fold_num, (_, val_idx) in enumerate(group_k_fold_splitter.split(X, y, groups=groups)):
        df_output.iloc[val_idx, df_output.columns.get_loc(fold_col)] = fold_num

    # check fold column values if they are in 0-(n_splits - 1) since their initial values were -1
    assert (df_output[fold_col] >= 0).all()

    if autosave:
        df_output.to_csv(os.path.join(ARTIFACTS_DIR, 'processed', 'train_table_with_folds.csv'), index=False)

    return df_output

def check_site_has_single_fold(
        df: pd.DataFrame,
        *,
        site_col: str = "site",
        fold_col: str = "fold") -> None:
    """
    Validate that each site (group) is assigned to exactly one fold.

    This ensures GroupKFold constraints are satisfied: the same site must not
    appear across multiple folds. Raises ValueError if any site maps to more
    than one fold.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing site and fold columns.
    site_col : str, default="site"
        Column name representing the grouping key (e.g., camera trap site).
    fold_col : str, default="fold"
        Column name representing the assigned fold id.

    Raises
    ------
    ValueError
        If at least one site is assigned to multiple folds.
    """

    n_folds_per_site = df.groupby(site_col)[fold_col].nunique()
    bad = n_folds_per_site[n_folds_per_site != 1]
    if len(bad) > 0:
        raise ValueError(f"Some sites appear in multiple folds: {bad.head(10).to_dict()}")

def check_no_site_overlap_between_train_valid(df: pd.DataFrame, *, site_col: str = "site", fold_col: str = "fold") -> None:
    """
    Validate there is no overlap of sites between train and validation splits per fold.

    For each fold f, the validation set is rows where fold_col == f and the training set
    is rows where fold_col != f. This function checks that the set of site values in
    validation does not intersect with the set of site values in training. Raises
    ValueError if any overlap is found.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing site and fold columns.
    site_col : str, default="site"
        Column name representing the grouping key (e.g., camera trap site).
    fold_col : str, default="fold"
        Column name representing the assigned fold id.

    Raises
    ------
    ValueError
        If any fold has at least one site present in both train and validation.
    """

    folds = sorted(df[fold_col].unique())
    for f in folds:
        train_sites = set(df.loc[df[fold_col] != f, site_col])
        valid_sites = set(df.loc[df[fold_col] == f, site_col])
        inter = train_sites & valid_sites
        if inter:
            raise ValueError(f"Site overlap detected in fold {f}: {list(inter)[:10]}")
