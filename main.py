import os
import tarfile
import urllib.request

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion

from scipy.stats import randint


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
DATA_PATH = 'datasets/housing'
DATA_FILE = 'housing.csv'
DATA_URL = DOWNLOAD_ROOT + DATA_PATH + '/housing.tgz'

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


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


def pre_process(data):
    cat_attributes = ['ocean_proximity']
    num_attributes = list(data)
    num_attributes.remove('ocean_proximity')

    # Imputer pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('imputer', Imputer(strategy='median')),
        ('attributes_adder', CombinedAttributesAdder()),
        ('normalization', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ('label_binarizer', LabelBinarizer()),
    ])

    pre_processing_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])

    return pre_processing_pipeline.fit_transform(data)


def hyper_parameter_search(X, Y):
    param_distributions = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

    regressor = RandomForestRegressor(n_estimators=130, max_features=7)

    rnd_search = RandomizedSearchCV(regressor, param_distributions=param_distributions,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error')
    rnd_search.fit(X, Y)
    return rnd_search.best_estimator_


def separate_labels(data, label):
    data_without_label = data.drop(label, axis=1)
    labels = data[label].copy()

    return data_without_label, labels


def test_model(model, test_set, test_set_labels):
    test_set_predictions = model.predict(test_set)
    test_set_mse = mean_squared_error(test_set_labels, test_set_predictions)
    return np.sqrt(test_set_mse)


def download_data(url, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        return False

    print('Downloading dataset...')
    tgz_path = os.path.join(path, "housing.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    tgz = tarfile.open(tgz_path)
    tgz.extractall(path=path)
    tgz.close()

    return True


def main():
    # Downloads dataset if it does not exist yet and loads it
    downloaded = download_data(DATA_URL, DATA_PATH)
    if downloaded:
        print('Dataset was downloaded...')

    data = load_data(DATA_PATH, DATA_FILE)

    # Stratified splitting
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
    train_set, test_set = strat_split_set(data, 'income_cat')
    train_set.drop('income_cat', axis=1, inplace=True)
    test_set.drop('income_cat', axis=1, inplace=True)

    # Extract labels
    housing, housing_labels = separate_labels(train_set, 'median_house_value')
    housing_test, housing_labels_test = separate_labels(test_set, 'median_house_value')

    # Pre process
    housing_prepared = pre_process(housing)
    housing_prepared_test = pre_process(housing_test)

    # Search for best regressor hyper-parameters
    best_regressor = hyper_parameter_search(housing_prepared, housing_labels)
    best_regressor.fit(housing_prepared, housing_labels)

    # Final test
    rmse = test_model(best_regressor, housing_prepared_test, housing_labels_test)
    print('RMSE: {}'.format(rmse))


if __name__ == '__main__':
    main()
