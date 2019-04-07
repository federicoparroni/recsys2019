from recommenders.recommender_base import RecommenderBase
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import data
import xgboost as xgb
from tqdm import tqdm


class XGBoost(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster'):
        name = 'xgboost'
        super(XGBoost, self).__init__(name=name, mode=mode, cluster=cluster)

        self.preds = None
        self.target_indices = data.target_indices(mode=mode, cluster=cluster)

    def fit(self):
        train = data.classification_train_df(mode=self.mode, cluster=self.cluster)
        X_train, y_train = train.iloc[:, 3:], train.iloc[:, 2]

        self.xg = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100)
        self.xg.fit(X_train, y_train)

    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        test = data.classification_test_df(mode=self.mode, cluster=self.cluster)
        X_test = test.iloc[:, 3:]

        preds = self.xg.predict_proba(X_test)
        self.preds = [a[1] for a in preds]

        current_index = 0
        test_df = data.test_df(mode=self.mode, cluster=self.cluster)

        predictions = []
        for index in tqdm(self.target_indices):
            impr = list(map(int, test_df.loc[index]['impressions'].split('|')))
            scores = self.preds[current_index:len(impr)]
            current_index = len(impr)

            sorted_impr = [impr for _, impr in sorted(zip(scores, impr))]
            predictions.append((index, sorted_impr))
        
        return predictions

if __name__ == '__main__':
    model = XGBoost(mode='small', cluster='no_cluster')
    model.evaluate()






