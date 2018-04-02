import pandas as pd
import numpy as np
from src.configs import *
from src.features.transform import categorical_to_ordinal
import collections


class HousePriceData:
    '''
    Load House Price data for Kaggle competition
    '''
    def __init__(self, train_path, test_path):
        self.trainset = pd.read_csv(train_path)
        self.testset = pd.read_csv(test_path)
        self.target = self.trainset["SalePrice"]

        self.workset = None
        self.train_id = self.trainset["Id"]
        self.test_id = self.testset["Id"]

        # Keep track of feature to use to train in class
        # Don'T really like that, but can't find a better way for now
        self.original_feature_name = list(self.trainset)
        self.created_feature = list()
        self.ignored_feature = list()
        self.usable_feature_name = list()

        self._create_join_workset()
        self._replace_significant_nan_values()
        self._replace_all_nan_values()
        self._change_col_type()
        self._apply_feature_engineering()

    def _replace_significant_nan_values(self, filler="MISSING"):
        for col in COL_WITH_NAN_SIGNIFICATION:
            self.workset[col] = self.workset[col].fillna(filler)

        self.update_train_and_test_set(self.workset)

    def get_usable_feature_name(self):
        self.usable_feature_name = self.original_feature_name

        for feature_name in self.created_feature:
            self.usable_feature_name.append(feature_name)

        for feature_name in self.ignored_feature:
            try:
                self.usable_feature_name.remove(feature_name)
            except Exception as e:
                pass

        return self.usable_feature_name

    def add_created_features(self, feature_names):
        if isinstance(feature_names, str):
            self.created_feature.append(feature_names)
        elif isinstance(feature_names, collections.Iterable):
            for i in feature_names:
                self.created_feature.append(i)
        else:
            raise ValueError('Not valid feature_names type. Should be str or Iterable')

    def add_ignore_features(self, feature_names):
        if isinstance(feature_names, str):
            self.ignored_feature.append(feature_names)
        elif isinstance(feature_names, collections.Iterable):
            for i in feature_names:
                self.ignored_feature.append(i)
        else:
            raise ValueError('Not valid feature_names type. Should be str or Iterable')

    def _replace_all_nan_values(self, filler="NO"):
        self.workset = self.workset.fillna(filler)
        self.update_train_and_test_set(self.workset)

    def _create_join_workset(self):
        self.workset = pd.concat([self.trainset[ORIGINAL_FEATURE_COLS], self.testset[ORIGINAL_FEATURE_COLS]], axis=0)

    def update_train_and_test_set(self, workset):
        self.trainset = workset[workset['Id'].isin(self.train_id)]
        self.testset = workset[workset['Id'].isin(self.test_id)]

    def update_workset(self, workset):
        self.workset = workset
        self.update_train_and_test_set(self.workset)

    def _change_col_type(self):
        for col in ["MoSold", "MSSubClass"]:
            self.workset[col] = self.workset[col].astype('str')

        self.update_train_and_test_set(self.workset)

    def _apply_feature_engineering(self):
        self.update_train_and_test_set(self.workset)

