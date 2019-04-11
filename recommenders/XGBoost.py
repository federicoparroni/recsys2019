import math
import xgboost as xgb
import data
from recommenders.recommender_base import RecommenderBase
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sps
tqdm.pandas()


class XGBoostWrapper(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.3, min_child_weight=1, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        name = 'xgboost'
        super(XGBoostWrapper, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.preds = None
        self.target_indices = data.target_indices(mode=mode, cluster=cluster)
        self.xg = xgb.XGBClassifier(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'learning_rate': (0.01, 0.3),
                                     'min_child_weight': (0, 1),
                                     'max_depth': (3, 20),
                                     'subsample': (0.5, 1),
                                     'colsample_bytree': (0.5, 1),
                                     'reg_lambda': (0, 1),
                                     'reg_alpha': (0, 1)
                                     }

    def fit(self):
        train = data.classification_train_df(
            mode=self.mode, sparse=True, cluster=self.cluster)
        X_train, y_train = train.iloc[:, 3:], train.iloc[:, 2]

        X_train = sps.csr_matrix(X_train.values)
        y_train = y_train.to_dense()

        self.xg.fit(X_train, y_train)

    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        test = data.classification_test_df(
            mode=self.mode, sparse=False, cluster=self.cluster)
        test = test.set_index(['session_id'])
        test_df = data.test_df(mode=self.mode, cluster=self.cluster)

        predictions = []
        for index in tqdm(self.target_indices):
            tgt_row = test_df.loc[index]
            tgt_sess = tgt_row['session_id']
            tgt_user = tgt_row['user_id']

            tgt_test = test.loc[[tgt_sess]]
            tgt_test = tgt_test[tgt_test['user_id'] == tgt_user]
            
            tgt_test = tgt_test.sort_values('impression_position')

            X_test = tgt_test.iloc[:, 2:]
            preds = self.xg.predict_proba(sps.csr_matrix(X_test.values))
            scores = [a[1] for a in preds]

            impr = list(map(int, tgt_row['impressions'].split('|')))
            scores_impr = [[scores[i], impr[i]] for i in range(len(impr))]
            scores_impr.sort(key=lambda x: x[0], reverse=True)

            predictions.append((index, [x[1] for x in scores_impr]))


        return predictions


if __name__ == '__main__':
    model = XGBoostWrapper(mode='small', cluster='no_cluster')
    model.evaluate(send_MRR_on_telegram=False)
