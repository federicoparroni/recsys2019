import math
import xgboost as xgb
import data
from recommenders.recommender_base import RecommenderBase
import numpy as np
import pandas as pd
from pandas import Series
from tqdm import tqdm
import scipy.sparse as sps
tqdm.pandas()


class XGBoostWrapper(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.3, min_child_weight=1, n_estimators=100, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        name = 'xgboost'
        super(XGBoostWrapper, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.preds = None
        self.scores_batch = None
        self.target_indices = data.target_indices(mode=mode, cluster=cluster)
        self.xg = xgb.XGBRanker(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha, n_jobs=-1, objective='rank:pairwise')

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'learning_rate': (0.01, 0.3),
                                     'max_depth': (2, 6),
                                     'n_estimators': (50, 500),
                                     'reg_lambda': (0, 1),
                                     'reg_alpha': (0, 1)
                                     }

    def fit(self):
        train = data.classification_train_df(
            mode=self.mode, sparse=True, cluster=self.cluster)
        train = train.sort_values(by=['user_id', 'session_id'])
        X_train, y_train = train.iloc[:, 3:], train.iloc[:, 2]
        X_train = X_train.to_coo().tocsr()
        
        group = np.load('/home/giovanni/Desktop/recsys2019/dataset/preprocessed/no_cluster/small/group.npy')
        print('data for train ready')

        self.xg.fit(X_train, y_train.to_dense(), group)
        print('fit done')

    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        test = data.xgboost_test_df(
            mode=self.mode, sparse=True)
        test_scores = test[['user_id', 'session_id',
                            'impression_position']].to_dense()

        # build aux dictionary
        d = {}
        for idx, row in test_scores.iterrows():
            sess_id = row['session_id']
            if sess_id in d:
                d[sess_id] += [idx]
            else:
                d[sess_id] = [idx]

        test_df = data.test_df(mode=self.mode, cluster=self.cluster)

        X_test = test.iloc[:, 3:-1]
        X_test = X_test.to_coo().tocsr()

        print('data for test ready')

        scores = list(self.xg.predict(X_test))
        test_scores['scores'] = Series(scores, index=test_scores.index)

        predictions = []
        self.scores_batch = []
        for index in tqdm(self.target_indices):

            # Get only test rows with same session&user of target indices
            tgt_row = test_df.loc[index]
            tgt_sess = tgt_row['session_id']
            tgt_user = tgt_row['user_id']

            tgt_test = test_scores.loc[d[tgt_sess]]
            tgt_test = tgt_test[tgt_test['user_id'] == tgt_user]

            tgt_test = tgt_test.sort_values('impression_position')

            scores = tgt_test['scores'].values

            impr = list(map(int, tgt_row['impressions'].split('|')))
            scores_impr = [[scores[i], impr[i]] for i in range(len(impr))]
            scores_impr.sort(key=lambda x: x[0], reverse=True)

            preds = [x[1] for x in scores_impr]
            predictions.append((index, preds))

            scores = [x[0] for x in scores_impr]
            self.scores_batch.append((index, preds, scores))

        return predictions


if __name__ == '__main__':
    model = XGBoostWrapper(mode='small', cluster='no_cluster')
    model.evaluate(send_MRR_on_telegram=True)
