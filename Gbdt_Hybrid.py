from recommenders.recommender_base import RecommenderBase
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
tqdm.pandas()
from pathlib import Path
import os
import data
import utils.functions as f
import xgboost as xgb
from preprocess_utils.dataset_xgboost import create_groups

"""
0.6670210196979507 learn_rate = 0.3
0.6669933320555111 learn_rate = 0.1
0.6671491436662834
"""

class Gbdt_Hybrid(RecommenderBase):

    """To create dataset, put submissions on submissions
                                                |__local
        At the moment, it takes the position in the recommended impressions of each submission for each item such as:
        user_id | session_id | item_id | item_pos_catboost | item_pos_xgboost | ...
        Then Xgboost is used to train and recommend. Since only local submissions can be used (thus the dataset size is like the local test size),
        in order to train the model and optimize hyperparameters, we need to cross-validate the dataset
    """

    def __init__(self, mode='local', learning_rate=0.3, min_child_weight=1, n_estimators=100, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        name = 'gbdt_hybrid'
        cluster = 'no_cluster'
        super(Gbdt_Hybrid, self).__init__(mode, cluster, name)

        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', '..', 'submissions/hybrid')
        self.gt_csv = self.data_directory.joinpath('ground_truth.csv')
        self.mode = mode

        directory = self.data_directory.joinpath(self.mode)

        self.xg = xgb.XGBRanker(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            n_jobs=-1, objective='rank:pairwise')


        if os.path.isfile('/Users/claudiorussointroito/Documents/GitHub/recsysTrivago2019/submissions/gbdt_train.csv'):
            for root, dirs, files in os.walk(directory):
                file = files[0]
                f = pd.read_csv(directory.joinpath(file))
                self.get_groups(f)
                self.train = pd.read_csv(str(directory.joinpath('..', '..', 'gbdt_train.csv')))
            return

        print('Creating gbdt-like dataset...')

        num_file = 0
        #train = pd.DataFrame()
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    f = pd.read_csv(directory.joinpath(file))
                    print(file)
                    #t = self.convert_and_add_pos(f, str(file))
                    if num_file == 0:
                        self.get_groups(f)
                        t = self.convert_and_add_labels(f, str(file))
                        # inserire qui la divisione in K fold delle sessioni
                        # Attaccare la colonna label al dataset
                        self.train = t
                    else:
                        t = self.convert_and_add_pos(f, str(file))
                        print(t.columns)
                        self.train = pd.merge(self.train, t, on=['user_id', 'session_id', 'item_id', 'step','timestamp'])
                    print(f'file read: {num_file}')
                    print(self.train.shape)
                num_file += 1

        self.train = self.train.sort_values(by=['trg_idx', 'impression_position'])
        self.train.to_csv('/Users/claudiorussointroito/Documents/GitHub/recsysTrivago2019/submissions/gbdt_train.csv', index=False)
        print('Dataset created!')
        print(self.train)
        print(self.train.columns)

    def expand_item_recommendations(self, df):
        res_df = df.copy()
        res_df.item_recommendations = res_df.item_recommendations.str.split(' ')
        res_df = res_df.reset_index()
        res_df = pd.DataFrame({
            col: np.repeat(res_df[col].values, res_df.item_recommendations.str.len())
            for col in res_df.columns.drop('item_recommendations')}
        ).assign(**{'item_recommendations': np.concatenate(res_df.item_recommendations.values)})[res_df.columns]

        res_df = res_df.rename(columns={'item_recommendations': 'item_id'})
        res_df = res_df.astype({'item_id': 'int'})
        return res_df[['index', 'user_id', 'session_id', 'item_id', 'step', 'timestamp']]

    def convert_and_add_pos(self, df, name=''):
        print('Convert and adding submission impression positions..')
        df = df.sort_values(by=['user_id', 'session_id'], axis=0)
        df_t = self.expand_item_recommendations(df)
        df['item_recommendations'] = df['item_recommendations'].apply(lambda x: x.split(' '))
        df = df.drop(['timestamp', 'step', 'frequence'], axis=1)
        df['index'] = df.index
        df = pd.merge(df_t, df, how='left', on=['index', 'user_id', 'session_id'])

        #df['item_pos_' + name] = df.progress_apply(lambda x: (x['item_recommendations'].index(str(x['item_id']))) + 1, axis=1)
        df['item_pos_' + name] = self.get_pos(df['item_id'].values, df['item_recommendations'].values)
        df = df.drop(['item_recommendations', 'index'], axis=1)
        return df

    def convert_and_add_labels(self, df, name=''):
        full = data.full_df()
        target_indices = data.target_indices(mode='local', cluster='no_cluster')

        full = full.loc[target_indices]
        full['trg_idx'] = list(full.index)
        full['impressions'] = full['impressions'].str.split('|')
        full = full[['user_id', 'session_id', 'reference', 'trg_idx', 'impressions']]
        df = pd.merge(df, full, on=['user_id', 'session_id'])
        df = self.convert_and_add_pos(df, name)
        print('Adding labels..')
        print(df.columns)

        #df['label'] = df.progress_apply(lambda x: 1 if str(x['reference']) == str(x['item_id']) else 0, axis=1)
        df['label'] = np.where(df['reference'].astype(str) == df['item_id'].astype(str), 1, 0)

        df['impression_position'] = self.get_pos(df['item_id'].values, df['impressions'].values)
        df = df.drop(['reference', 'impressions'], axis=1)
        return df

    def get_pos(self, item, rec):
        res = np.empty(item.shape)
        for i in range(len(item)):
            res[i] = rec[i].index(str(item[i])) + 1
        return res.astype(int)

    def get_groups(self, df):
        print('Getting groups for k-fold..')
        sessions = list(set(list(df['session_id'].values)))
        sess_dict = {}
        for i in range(len(sessions)):
            sess_dict[i] = sessions[i]

        sess_keys = list(sess_dict.keys())
        groups = []
        for i in range(4):
            group = list(np.random.choice(sess_keys, 30627, replace=False))
            groups.append(group)
            sess_keys = list(set(sess_keys) - set(group))

        groups.append(sess_keys)

        self.sess_groups = []
        for j in range(5):
            sess_group = []
            for i in groups[j]:
                sess_group.append(sess_dict[i])
            self.sess_groups.append(sess_group)
        print('Done!')


    def fit(self, train):
        X_train = train.drop(
            ['user_id', 'session_id', 'item_id', 'label', 'timestamp', 'step', 'trg_idx'], axis=1)
        print(f'Train columns: {list(X_train.columns)}')
        y_train = list(train['label'].values)
        X_train = X_train.astype(np.float64)
        group = create_groups(train)

        self.xg.fit(X_train, y_train, group)


    def recommend_batch(self, test, target_indices):

        X_test = test.sort_values(by=['trg_idx', 'impression_position'])
        X_test = X_test.drop(['user_id', 'session_id', 'item_id', 'label', 'timestamp', 'step', 'trg_idx'], axis=1)
        X_test = X_test.astype(np.float64)
        full_impressions = data.full_df()
        print('data for test ready')
        scores = list(self.xg.predict(X_test))
        final_predictions = []
        count = 0
        for index in tqdm(target_indices):
            impressions = list(
                map(int, full_impressions.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            _, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr)))
            count = count + len(impressions)
        return final_predictions

    def get_scores_batch(self):
        pass

    def cross_validation(self):
        self.mrrs = []
        for i in range(len(self.sess_groups)):
            mask = self.train['session_id'].isin(self.sess_groups[i])
            test_df = self.train[mask]
            train_df = self.train[~mask]
            target_indices = list(set(test_df['trg_idx'].values))
            target_indices.sort()

            self.fit(train_df)
            recs = self.recommend_batch(test_df, target_indices)
            mrr = self.compute_MRR(recs)
            self.mrrs.append(mrr)

        print(f'Averaged MRR: {sum(self.mrrs)/len(self.mrrs)}')


if __name__=='__main__':
    model = Gbdt_Hybrid(mode='local')
    model.cross_validation()
