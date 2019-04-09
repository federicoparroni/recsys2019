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

        self.xg = xgb.XGBClassifier()
        self.xg.fit(X_train, y_train)

    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        test = data.classification_test_df(mode=self.mode, cluster=self.cluster)
        print(test)
        test_df = data.test_df(mode=self.mode, cluster=self.cluster)
    
        predictions = []
        for index in tqdm(self.target_indices):
            tgt_row = test_df.loc[index]
            tgt_user = tgt_row['user_id']
            tgt_sess = tgt_row['session_id']
            tgt_test = test[(test['user_id'] == tgt_user) & (test['session_id'] == tgt_sess)]
            tgt_test = tgt_test.sort_values(['impression_position'])
            X_test = tgt_test.iloc[:, 3:]
            
            preds = self.xg.predict_proba(X_test)
            scores = [a[1] for a in preds]
            impr = list(map(int, tgt_row['impressions'].split('|')))

            scores_impr = [[scores[i], impr[i]] for i in range(len(impr))]
            scores_impr.sort(key=lambda x: x[0], reverse=True)

            predictions.append((index, [x[1] for x in scores_impr]))
            
        return predictions

if __name__ == '__main__':
    model = XGBoost(mode='local', cluster='no_cluster')
    model.evaluate(send_MRR_on_telegram=True)
