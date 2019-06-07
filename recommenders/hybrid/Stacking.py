from recommenders.recommender_base import RecommenderBase
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
tqdm.pandas()
from pathlib import Path
import os
import data
import xgboost as xgb

class Stacking(RecommenderBase):

    def __init__(self, mode='local', learning_rate=0.3, min_child_weight=1, n_estimators=300, max_depth=3,
                 subsample=1, colsample_bytree=1, reg_lambda=1.0, reg_alpha=0):
        name = 'Stacking'
        cluster = 'no_cluster'
        super(Stacking, self).__init__(mode, cluster, name)

        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', '..', 'submissions/hybrid')
        # self.gt_csv = self.data_directory.joinpath('ground_truth.csv')
        self.mode = mode
        self.xg = xgb.XGBRanker(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            n_jobs=-1, objective='rank:ndcg')

    def create_groups_cv(self,df):
        df = df[['index', 'item_id']]
        group = df.groupby('index',
                           sort=False).apply(lambda x: len(x)).values
        return group

    def fit(self):
        train = pd.read_csv(str(self.data_directory.joinpath('..', 'gbdt_train.csv')))
        val = pd.read_csv(str(self.data_directory.joinpath('..', 'gbdt_test.csv')))

        X_train = train.drop(
            ['item_id', 'label', 'index'], axis=1)
        X_val = val.drop(['item_id', 'label', 'index'], axis=1)

        print(f'Train columns: {list(X_train.columns)}')
        y_train = list(train['label'].values)
        y_val = list(train['label'].values)

        X_train = X_train.astype(np.float64)
        X_val = X_val.astype(np.float64)

        group = self.create_groups_cv(train)
        group_val = self.create_groups_cv(val)

        print('Training XGBOOST..')
        self.xg.fit(X_train, y_train, group, eval_set=[
                    (X_val, y_val)], eval_group=[group_val], eval_metric='ndcg', early_stopping_rounds=200)

        print('Training done!')
