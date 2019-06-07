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
import utils.telegram_bot as HERA

class Gbdt_Hybrid(RecommenderBase):

    """To create dataset, put submissions on submissions
                                                |__local

        It takes the scores  of each submission for each item such as:
        user_id | session_id | item_id | item_score_catboost | item_score_xgboost | ...
        Then Xgboost is used to train and recommend. Since only local submissions can be used (thus the dataset size is like the local test size),
        in order to train the model and optimize hyperparameters, we need to cross-validate the dataset
    """

    """
    learning_rate=0.3, min_child_weight=1, n_estimators=300, max_depth=3,
                 subsample=1, colsample_bytree=1, reg_lambda=1.0, reg_alpha=0"""


    def __init__(self, mode='local', learning_rate=0.3, min_child_weight=1, n_estimators=300, max_depth=3,
                 subsample=1, colsample_bytree=1, reg_lambda=1.0, reg_alpha=0):
        name = 'gbdt_hybrid'
        cluster = 'no_cluster'
        super(Gbdt_Hybrid, self).__init__(mode, cluster, name)

        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', '..', 'submissions/hybrid')
        #self.gt_csv = self.data_directory.joinpath('ground_truth.csv')
        self.mode = mode
        self.full = data.full_df()

        self.local_target_indices = data.target_indices(mode='local', cluster='no_cluster')
        self.full_target_indices = data.target_indices(mode='full', cluster='no_cluster')

        directory = self.data_directory.joinpath('local')

        full_dir = self.data_directory.joinpath('full')

        self.xg = xgb.XGBRanker(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            n_jobs=-1, objective='rank:ndcg')


        self.cv_path = self.data_directory.joinpath('cross_validation')

        # if not os.path.isfile(str(directory.joinpath('..', '..', 'gbdt_test.csv'))):
        #     self.test_full = self.create_dataset(full_dir, indices=self.full_target_indices, mode='full',
        #                                          name='test')  # directory per il full--> dataset per il test

        # Get the group for the cross validation and load the train previously saved

        # if os.path.isfile(str(directory.joinpath('..', '..', 'gbdt_train.csv'))):
        #     for root, dirs, files in os.walk(directory):
        #         if str(files[0]) == '.DS_Store':
        #             file = files[1]
        #         else:
        #             file = files[0]
        #
        #         if file.endswith(".npy"):
        #             f = np.load(directory.joinpath(file))
        #             f = pd.DataFrame(f, columns=['index', 'item_recommendations', 'scores'])
        #             f = f[f['index'].isin(self.local_target_indices)]
        #
        #         elif file.endswith(".pickle"):
        #             f = pd.read_pickle(directory.joinpath(file))
        #             f = pd.DataFrame(f, columns=['index', 'item_recommendations', 'scores'])
        #             f = f[f['index'].isin(self.local_target_indices)]
        #
        #         else:
        #             f = pd.read_csv(directory.joinpath(file))
        #         self.get_groups(f)
        #         self.train = pd.read_csv(str(directory.joinpath('..', '..', 'gbdt_train.csv')))
        #     return
        #
        #
        # self.train = self.create_dataset(directory, self.local_target_indices)  #dataset per il train (local test)
        #
        # self.test_full = self.create_dataset(full_dir, indices=self.full_target_indices, mode='full', name='test') # directory per il full--> dataset per il test


    def create_dataset(self, directory, indices, mode='local', name='train'):
        print('Creating gbdt-like dataset...')

        num_file = 0
        #train = pd.DataFrame()
        for root, dirs, files in os.walk(directory):
            for file in files:
                print(f'Reading {str(file)}')

                if str(file) == '.DS_Store':
                    continue

                if file.endswith(".csv"):

                    f = pd.read_csv(directory.joinpath(file))

                    print(file)
                    #t = self.convert_and_add_pos(f, str(file))
                    if num_file == 0:
                        self.get_groups(f)

                        t = self.convert_and_add_labels(f, str(file))
                        self.train = t
                    else:
                        t = self.convert_and_add_pos(f, str(file))

                        print(t.columns)
                        self.train = pd.merge(self.train, t, on=['user_id', 'session_id', 'item_id', 'step','timestamp'])

                    print(f'file read: {num_file}')
                    print(self.train.shape)

                else:
                    if file.endswith(".npy"):

                        f = np.load(directory.joinpath(file))
                        f = pd.DataFrame(f, columns=['index', 'item_recommendations', 'scores'])
                        f = f[f['index'].isin(indices)]
                        f = f.astype({'index' : int})

                    elif file.endswith(".pickle"):
                        f = pd.read_pickle(directory.joinpath(file))
                        f = pd.DataFrame(f, columns=['index', 'item_recommendations', 'scores'])
                        f = f[f['index'].isin(indices)]

                    print(file)
                    if num_file == 0:
                        if mode=='local':
                            self.get_groups(f)

                        #TODO da togliere quando rnn avrà gli indice per bene
                        if (mode=='full') & (file == 'xgboost.npy'):
                            f = f[~f['index'].isin([16727760, 18888288])]
                        elif (mode=='full') & (file == 'rnn.npy'):
                            f = f[~f['index'].isin([16727762, 18888282])]

                        t = self.convert_and_add_labels(f, name=str(file), target_indices=indices, mode=mode)
                        df = t
                    else:
                        #TODO da togliere quando rnn avrà gli indice per bene
                        if (mode == 'full') & (file == 'xgboost.npy'):
                            f = f[~f['index'].isin([16727760, 18888288])]
                        elif (mode == 'full') & (file == 'rnn.npy'):
                            f = f[~f['index'].isin([16727762, 18888282])]

                        t = self.convert_and_assign_score(f, str(file))

                        print(t.columns)
                        df = pd.merge(df, t, on=['index', 'item_id'])

                    print(f'file read: {num_file}')
                    print(df.shape)


                num_file += 1
                if str(file) == '.DS_Store':
                    num_file -= 1

        df = df.sort_values(by=['index', 'impression_position'])
        df.to_csv(str(directory.joinpath('..', '..', 'gbdt_' + name + '.csv')), index=False)
        print('Dataset created!')
        print(df.columns)
        return df


    """  ------------- Second step re-ranking per scores ------------------ """
    """
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
        full = self.full
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
    """


    """  ------------- Second step re-ranking per scores ------------------ """


    def expand_item_recommendations(self, df, perScore=True):
        res_df = df.copy()

        if perScore == False:
            res_df.item_recommendations = res_df.item_recommendations.str.split(' ')

        res_df = res_df.reset_index()
        res_df = pd.DataFrame({
            col: np.repeat(res_df[col].values, res_df.item_recommendations.str.len())
            for col in res_df.columns.drop('item_recommendations')}
        ).assign(**{'item_recommendations': np.concatenate(res_df.item_recommendations.values)})[res_df.columns]

        res_df = res_df.rename(columns={'item_recommendations': 'item_id'})
        res_df = res_df.astype({'item_id': 'int'})
        return res_df[['index', 'item_id']]


    def convert_and_add_labels(self, df, target_indices, name='', mode='local'):
        #full = data.full_df()
        full = self.full
        #target_indices = data.target_indices(mode=mode, cluster='no_cluster')

        full = full.loc[target_indices]
        full['index'] = list(full.index)
        full['impressions'] = full['impressions'].str.split('|')
        full = full[['reference', 'index', 'impressions']]

        #df = pd.merge(df, full, on=['user_id', 'session_id'])
        df = pd.merge(df, full, on=['index'], how='left')

        df = self.convert_and_assign_score(df, name)

        print('Adding labels..')
        print(df.columns)

        # df['label'] = df.progress_apply(lambda x: 1 if str(x['reference']) == str(x['item_id']) else 0, axis=1)

        df['label'] = np.where(df['reference'].astype(str) == df['item_id'].astype(str), 1, 0)

        df['impression_position'] = self.get_pos(df['item_id'].values, df['impressions'].values)

        df = df.drop(['reference', 'impressions'], axis=1)
        return df

    def convert_and_assign_score(self, df, name):
        print('Convert and adding submission scores positions..')
        df_t = self.expand_item_recommendations(df)

        df = pd.merge(df_t, df, on=['index'], how='left')
        df['score_' + name] = self.get_score(df['item_id'].values, df['scores'].values, df['item_recommendations'].values)
        df = df.drop(['scores', 'item_recommendations'], axis=1)
        return df


    def get_score(self, item, scores, rec):
        res = np.empty(item.shape)
        for i in range(len(item)):
            res[i] = scores[i][rec[i].index(item[i])]
        return res

    def get_pos(self, item, rec):
        res = np.empty(item.shape)
        for i in range(len(item)):
            res[i] = rec[i].index(str(item[i])) + 1

        return res.astype(int)

    def get_groups(self, df):
        print('Getting groups for k-fold..')
        if 'session_id' in df.columns:
            sessions = list(set(list(df['session_id'].values)))
        else:
            sessions = list(set(list(df['index'].values)))

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


    """-----------------------------    CROSS-VALIDATION --------------------------------"""

    def create_groups_cv(self,df):
        df = df[['index', 'item_id']]
        group = df.groupby('index',
                           sort=False).apply(lambda x: len(x)).values
        return group

    def fit_cv(self, train, test):
        #X_train = train.drop(['user_id', 'session_id', 'item_id', 'label', 'timestamp', 'step', 'trg_idx'], axis=1)
        X_train = train.drop(['item_id', 'label', 'index'], axis=1)

        print(f'Train columns: {list(X_train.columns)}')
        y_train = list(train['label'].values)
        X_train = X_train.astype(np.float64)
        group = self.create_groups_cv(train)

        """
        # validation test
        X_test = test.sort_values(by=['index', 'impression_position'])
        X_test = X_test.drop(['item_id', 'label', 'index'], axis=1)
        X_test = X_test.astype(np.float64)
        y_test = list(test['label'].values)
        test_group = self.create_groups_cv(test)
        """

        print('Training XGBOOST..')
        self.xg.fit(X_train, y_train, group)

        #self.xg.fit(X_train, y_train, group, eval_set=[
        #            (X_test, y_test)], eval_group=[test_group], eval_metric='ndcg', verbose=True, early_stopping_rounds=50)
        print('Training done!')

    def recommend_batch_cv(self, test, target_indices):

        X_test = test.sort_values(by=['index', 'impression_position'])
        X_test = X_test.drop(['item_id', 'label', 'index'], axis=1)
        X_test = X_test.astype(np.float64)
        full_impressions = self.full
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

    def cross_validation(self):
        self.mrrs = []
        self.folds = []
        for i in range(len(self.sess_groups)):
            if os.path.isfile(str(self.cv_path) + '/train' + str(i) + '.csv'):
                print(f'Getting train and test number: {i}')
                path = str(self.cv_path) + '/train' + str(i) + '.csv'
                train_df = pd.read_csv(str(self.cv_path) + '/train' + str(i) + '.csv')
                test_df = pd.read_csv(str(self.cv_path) + '/test' + str(i) + '.csv')


                """Create folds: five tuples (in, out) where in is a list of indices to be used as the training samples for the n th fold
                 and out is a list of indices to be used as the testing samples for the n th fold. """
                #in_idx = list(train_df.index)
                #out_idx = list(test_df.index)
                #self.folds.append((in_idx, out_idx))
            else:
                if 'session_id' in self.train.columns:
                    mask = self.train['session_id'].isin(self.sess_groups[i])
                else:
                    mask = self.train['index'].isin(self.sess_groups[i])

                test_df = self.train[mask]
                train_df = self.train[~mask]
                train_df.to_csv(str(self.cv_path) + '/train' + str(i) + '.csv', index=False)
                test_df.to_csv(str(self.cv_path) + '/test' + str(i) + '.csv', index=False)


            #target_indices = list(set(test_df['trg_idx'].values))
            target_indices = list(set(test_df['index'].values))

            target_indices.sort()

            self.fit_cv(train_df, test_df)
            recs = self.recommend_batch_cv(test_df, target_indices)
            mrr = self.compute_MRR(recs)
            self.mrrs.append(mrr)

        print(f'Averaged MRR: {sum(self.mrrs)/len(self.mrrs)}')
        return sum(self.mrrs)/len(self.mrrs)

    """----------------------------------------------------------------------------------------------------------"""


    def fit(self):
        train = pd.read_csv(str(self.data_directory.joinpath('..','gbdt_train.csv')))
        X_train = train.drop(
            ['item_id', 'label', 'index'], axis=1)
        print(f'Train columns: {list(X_train.columns)}')
        y_train = list(train['label'].values)
        X_train = X_train.astype(np.float64)

        group = self.create_groups_cv(train)

        print('Training XGBOOST..')
        self.xg.fit(X_train, y_train, group)
        print('Training done!')

    def recommend_batch(self):
        test = pd.read_csv(str(self.data_directory.joinpath('..','gbdt_test.csv')))
        target_indices = data.target_indices(mode='full')
        #target_indices = sorted(target_indices)

        # Take indices in common between test and target_indices
        test_idx = set(list(test['index'].values))
        print(f'Length test_idx : {len(test_idx)}')
        print(f'Length target_indices : {len(target_indices)}')
        target_indices = set(target_indices) & test_idx
        print(f'Length indices in common: {len(target_indices)}')
        target_indices = sorted(target_indices)
        #target_indices = sorted(test_idx)

        X_test = test.sort_values(by=['index', 'impression_position'])
        X_test = X_test.drop(['item_id', 'label', 'index'], axis=1)
        X_test = X_test.astype(np.float64)
        # full_impressions = data.full_df()
        full_impressions = self.full
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


if __name__=='__main__':
    model = Gbdt_Hybrid(mode='local')
    #model.evaluate()
    model.run()
