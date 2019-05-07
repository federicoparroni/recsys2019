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
        name = 'xgboost_ranker'
        super(XGBoostWrapper, self).__init__(
            name=name, mode=mode, cluster=cluster)

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
        X_train, y_train, group = data.dataset_xgboost_train(mode=self.mode, cluster=self.cluster)
        print('data for train ready')

        self.xg.fit(X_train, y_train, group)
        print('fit done')

    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        X_test, target_indices_reordered = data.dataset_xgboost_test(mode=self.mode, cluster=self.cluster)
        full_impressions = pd.read_csv('dataset/preprocessed/full.csv', usecols=["impressions"])
        print('data for test ready')

        scores = list(self.xg.predict(X_test))

        final_predictions = []
        count = 0
        for index in tqdm(target_indices_reordered):
            impressions = list(map(int, full_impressions.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            if len(predictions) == 0:
                print('a')
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            _, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr)))
            count = count+len(impressions)
        return final_predictions


if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    model = XGBoostWrapper(mode=mode, cluster='no_cluster')
    #model.evaluate(send_MRR_on_telegram=True)
    model.run()
