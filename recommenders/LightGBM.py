import data
import math
from recommenders.recommender_base import RecommenderBase
import pandas as pd
from pandas import Series
from tqdm import tqdm
tqdm.pandas()
import lightgbm as lgb
from lightgbm import plot_importance
from sklearn import preprocessing

# On small dataset: 0.6505083796837017

class LightGBM(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.3, min_child_weight=1, n_estimators=100, max_depth=5,
                 num_leaves = 31,
                 subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0, feature_fraction = 1, bagging_fraction = 1, bagging_freq = 0):
        name = 'lightgbm'
        super(LightGBM, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.preds = None
        self.scores_batch = None
        self.target_indices = data.target_indices(mode=mode, cluster=cluster)
        self.lgb = lgb.LGBMClassifier(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            n_jobs=-1, feature_fraction= feature_fraction, bagging_fraction= bagging_fraction, bagging_freq= bagging_freq,
            num_leaves=num_leaves)

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
            mode=self.mode, sparse=False, cluster=self.cluster, algo='lightGBM')

        X_train, y_train = train.iloc[:, 3:], train.iloc[:, 2]

        X_train, cat_features = self.handle_cat_features(X_train, 'device', 'kind_action_reference_appeared_last_time')

        # TODO add city as feature

        print('data for train ready')

        self.lgb.fit(X_train, y_train.to_dense(), categorical_feature=cat_features)
        print('fit done')

    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        test = data.classification_test_df(
            mode=self.mode, sparse=False, cluster=self.cluster, algo='lightGBM')

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

        X_test = test.iloc[:, 3:]
        X_test, cat_features = self.handle_cat_features(X_test, 'device', 'kind_action_reference_appeared_last_time')
        # X_test = X_test.to_coo().tocsr()

        print('data for test ready')

        preds = self.lgb.predict_proba(X_test)
        scores = [a[1] for a in preds]
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

    def handle_cat_features(self, df, *argv):
        categorical_features = []
        features = []
        num_of_columns = df.shape[1]

        le = preprocessing.LabelEncoder()

        for arg in argv:
            categorical_features.append(arg)

        for i in range(0, num_of_columns):
            column_name = df.columns[i]
            column_type = df[column_name].dtypes

            if column_name in categorical_features:
                le.fit(df[column_name])
                feature_classes = list(le.classes_)
                encoded_feature = le.transform(df[column_name])
                df[column_name] = pd.Series(encoded_feature, dtype='int')
                continue
            df[column_name] = df[column_name].astype(int)


        return df, categorical_features

    def plot_importance(self):
        plot_importance(self.lgb)
