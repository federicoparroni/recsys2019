from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
import pandas as pd
import data
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
import numpy as np
from recommenders.recommender_base import RecommenderBase
import os
from pathlib import Path
from tqdm import tqdm
from preprocess_utils.dataset_lightGBM import create_dataset


class LightGBMRanker(RecommenderBase):

    def __init__(self,
                 #class_train = pd.DataFrame(), class_test=pd.DataFrame(), new_target_indices = None, test = pd.DataFrame(),
                 mode='small', cluster='no_cluster', learning_rate=0.3,
                 min_child_weight=1, n_estimators=800, max_depth=2,
                 subsample=1, colsample_bytree=1, subsample_freq=0, reg_lambda=0, reg_alpha=0.0,
                 tree_learner = 'feature'
                 ):
        name = 'lightgbmRanker'
        super(LightGBMRanker, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.curr_dir = Path(__file__).absolute().parent
        self.data_dir = self.curr_dir.joinpath('..', 'dataset/preprocessed/{}/{}/lightGBM'.format(cluster, mode))
        self.mode = mode
        self.cluster = cluster
        self.algo = 'lightGBM'

        self.model = lgb.LGBMRanker(learning_rate=learning_rate, min_child_weight=min_child_weight, n_estimators=n_estimators, max_depth=max_depth,
                      subsample=subsample, colsample_bytree=colsample_bytree, subsample_freq=subsample_freq,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                        objective='lambdarank', max_position=1, importance_type='split',
                                    #num_threads=2,
                                    #tree_learner='feature'
                                    )

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
        }

        self.hyperparameters_dict = {'learning_rate': (0.01, 0.3),
                                     'max_depth': (2, 6),
                                     'n_estimators': (50, 500),
                                     'reg_lambda': (0, 1),
                                     'reg_alpha': (0, 1)
                                     }

    def fit(self):

        svm_filename = 'svmlight_train.txt'
        _path = self.data_dir.joinpath(svm_filename)

        if os.path.isfile(str(_path)):
            self.X_train, self.y_train = load_svmlight_file( str(_path))
            self.q_train = np.loadtxt( str(_path) + '.query')

        self.model.fit(self.X_train, self.y_train, group=self.q_train,
          # eval_set=[(X_test, y_test)], eval_group=[q_test],
            )


    def recommend_batch(self):

        svm_filename = 'svmlight_test.txt'
        _path = self.data_dir.joinpath(svm_filename)

        X_test, y_test = load_svmlight_file(str(_path))

        target_indices = data.target_indices(self.mode, self.cluster)
        target_indices.sort()

        test = data.test_df('small', 'no_cluster')
        print('data for test ready')

        scores = list(self.model.predict(X_test))

        final_predictions = []
        count = 0
        for index in tqdm(target_indices):
            impressions = list(map(int, test.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            _, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr)))
            count = count + len(impressions)

        return final_predictions

    def get_scores_batch(self):
        pass

    def handle_cat_features(self, df, *argv):

        categorical_features = []
        num_of_columns = df.shape[1]

        le = preprocessing.LabelEncoder()

        for arg in argv:
            categorical_features.append(arg)

        for i in range(0, num_of_columns):
            column_name = df.columns[i]
            #column_type = df[column_name].dtypes

            if column_name in categorical_features:
                le.fit(df[column_name])
                encoded_feature = le.transform(df[column_name])
                df[column_name] = pd.Series(encoded_feature+1, dtype='category')
                continue
            elif column_name in ['user_id', 'session_id']:
                continue
            try:
                df[column_name] = df[column_name].astype(int)
            except:
                print(f'{column_name} contains NaN values')
                df.loc[:, column_name] = df[column_name].fillna(0).astype(int)


        return df, categorical_features

if __name__ == '__main__':
    model = LightGBMRanker(mode='small', cluster='no_cluster')
    model.evaluate(send_MRR_on_telegram=False)
