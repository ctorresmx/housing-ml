import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt

DATA_PATH = "datasets/housing"
DATA_FILE = "housing.csv"


def load_data(path, file):
    csv_path = os.path.join(path, file)
    return pd.read_csv(csv_path)


def strat_split_set(data, label):
    stratified_suffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                                     random_state=42)

    for train_index, test_index in stratified_suffle_split.split(data, data[label]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    return strat_train_set, strat_test_set


def main():
    # Categorizing the median income for stratified splitting
    data = load_data(DATA_PATH, DATA_FILE)
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)

    # Applying stratified set splitting
    train_set, test_set = strat_split_set(data, 'income_cat')
    train_set.drop('income_cat', axis=1, inplace=True)
    test_set.drop('income_cat', axis=1, inplace=True)


if __name__ == '__main__':
    main()
