import math
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import utils.telegram_bot as HERA
import data
from recommenders.recommender_base import RecommenderBase
import numpy as np
import pandas as pd
from pandas import Series
from tqdm import tqdm
import scipy.sparse as sps
tqdm.pandas()


class XGBoostWrapperClassifier(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster', learning_rate=1, min_child_weight=1, n_estimators=100, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        name = 'xgboost_classifier'
        super(XGBoostWrapperClassifier, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.xgb = xgb.XGBClassifier(silent=False,
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
                                     'n_estimators': (300, 1000),
                                     'reg_lambda': (0, 1),
                                     'reg_alpha': (0, 1)
                                     }

    def fit(self):
        train_df = data.dataset_xgboost_classifier_train(self.mode, self.cluster)
        train_df = train_df.drop(["user_id", "session_id"], axis=1)
        X_train = train_df.drop("label", axis=1)
        Y_train = train_df["label"]
        self.xgb.fit(X_train, Y_train)

    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        test_df = data.dataset_xgboost_classifier_test(self.mode, self.cluster)
        temp = test_df.copy()
        test_df = test_df.drop(["user_id", "session_id"], axis=1)
        X_test = test_df.drop("label", axis=1)
        Y_test = test_df["label"]


        Y_pred = self.xgb.predict(X_test)
        #temp["label"] = Y_pred
        #temp = temp[["user_id", "session_id", "label"]]
        #temp.to_csv("classifier.csv")
        return Y_test, Y_pred

    def evaluate(self, send_MRR_on_telegram = False):
        self.fit()
        print(self.xgb.feature_importances_)
        Y_test, Y_pred = self.recommend_batch()
        report = classification_report(Y_test , Y_pred)
        print(report)
        if send_MRR_on_telegram:
            HERA.send_message('evaluating classifier {} on {}.\n Classification report is: \n {}\n\n'.format(self.name, self.mode, report))

if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    model = XGBoostWrapperClassifier(mode=mode, cluster='no_cluster')
    model.evaluate(send_MRR_on_telegram=True)
    #model.run()
