from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
import pandas as pd
import data
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
import numpy as np
from recommenders.recommender_base import RecommenderBase
import os

class LightGBMRanker(RecommenderBase):

    def __init__(self, mode='small', cluster='no_cluster', learning_rate=0.08, min_child_weight=1, n_estimators=500, max_depth=5,
                 subsample=1, colsample_bytree=1, subsample_freq=0, reg_lambda=1, reg_alpha=0.1):
        name = 'lightgbmRanker'
        super(LightGBMRanker, self).__init__(
            name=name, mode=mode, cluster=cluster)
        self.mode = mode
        self.cluster = cluster
        self.algo = 'lightGBM'

        self.model = lgb.LGBMRanker(learning_rate=learning_rate, min_child_weight=min_child_weight, n_estimators=n_estimators, max_depth=max_depth,
                      subsample=subsample, colsample_bytree=colsample_bytree, subsample_freq=subsample_freq,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                        objective='lambdarank', max_position=25, importance_type='split', two_round=True)

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

        if os.path.isfile('../dataset/preprocessed/no_cluster/local/LambdaRank_dataset/svmlight_train.txt'):
            self.X_train, self.y_train = load_svmlight_file(
                '../dataset/preprocessed/no_cluster/local/LambdaRank_dataset/svmlight_train.txt')
            self.q_train = np.loadtxt(
             '../dataset/preprocessed/no_cluster/local/LambdaRank_dataset/svmlight_train.txt.query')

        else:
            #train = data.classification_train_df(
            #   mode=self.mode, sparse=False, cluster=self.cluster, algo='lightGBM')
            path = '../dataset/preprocessed/{}/{}/classification_train_{}.csv'.format(
                self.cluster, self.mode, self.algo)


            train = pd.read_csv(path)
            print('Ha letto il dataset')
            train = train.drop(['Unnamed: 0', 'Unnamed: 0.1', 'filters_when_clickout'], axis=1)
            print('Eliminate due colonne')
            train, cat = self.handle_cat_features(train, 'device', 'kind_action_reference_appeared_last_time', 'sort_order_active_when_clickout')
            print('Ha finito handle cat features')
            train.to_csv('../dataset/preprocessed/no_cluster/local/dataset_preprocessed.csv')


            train = pd.read_csv('../dataset/preprocessed/no_cluster/local/dataset_preprocessed.csv')
            self.to_queries_dataset(train)
            print('Convertito il dataset in query based')

            self.X_train, self.y_train = load_svmlight_file(
                '/Volumes/DR/Dataset TrivagoChallenge/dataset/preprocessed/no_cluster/local/LambdaRank_dataset/svmlight_train.txt')
            print('Ha caricato il dataset in formato svmlight')
            self.q_train = np.loadtxt(
                '/Volumes/DR/Dataset TrivagoChallenge/dataset/preprocessed/no_cluster/local/LambdaRank_dataset/svmlight_train.txt.query')

        self.model.fit(self.X_train, self.y_train, group=self.q_train,
          # eval_set=[(X_test, y_test)], eval_group=[q_test],
            )

    def func(self, x):
        #x is a groupd df containing same id (which now is the index)
        target_idx = x.trg_idx.values[0]

        X_test = x.drop(['label', 'trg_idx'], axis=1).values
        y_test = x['label'].values
        queries_test = x.trg_idx.values

        scores = self.model.predict(X_test, num_iteration=150)

        impr = list(map(int, self.test_df.at[target_idx, 'impressions'].split('|')))

        scores_impr = [[scores[i], impr[i]] for i in range(len(impr))]
        scores_impr.sort(key=lambda x: x[0], reverse=True)

        preds = [x[1] for x in scores_impr]
        scores = [x[0] for x in scores_impr]

        self.predictions.append((target_idx, preds))
        self.scores_batch.append((target_idx, preds, scores))


    def recommend_batch(self):
        test_df = data.classification_test_df(
            mode=self.mode, sparse=False, cluster=self.cluster, algo='lightGBM')

        test_df = test_df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'filters_when_clickout'], axis=1)

        test_df, cat = self.handle_cat_features(test_df, 'device', 'kind_action_reference_appeared_last_time', 'sort_order_active_when_clickout')
        test_df = test_df.sort_values(by=['user_id', 'session_id'])


        # set label column as first column
        cols = list(test_df.columns)
        cols.insert(0, cols.pop(cols.index('label')))
        test_df = test_df.loc[:, cols]
        #test_df = test_df.drop(['user_id', 'session_id'], axis=1)

        #test_df = test_df.drop(['device', 'kind_action_reference_appeared_last_time'], axis=1)

        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)
        print("loading test df")
        self.test_df = data.test_df(self.mode, self.cluster)

        sessi_target = self.test_df.loc[target_indices].session_id.values
        self.dict_session_trg_idx = dict(zip(sessi_target, target_indices))

        test_df['trg_idx'] = test_df.apply(lambda row: self.dict_session_trg_idx.get(row.session_id), axis=1)

        test_df.drop(['user_id', 'session_id'], inplace=True, axis=1)
        test_df = test_df.sort_values(by=['trg_idx', 'impression_position'])

        self.predictions = []
        self.scores_batch = []

        test_df.groupby('trg_idx', as_index=False).apply(self.func)

        return self.predictions

    def get_scores_batch(self):
        pass

    def to_queries_dataset(self, df, save_dataset=True, name='train'):
        print('Creating query-based dataset...')
        t = df.sort_values(by=['user_id', 'session_id'])
        t = t.assign(id=(t['user_id'] + '_' + t['session_id']).astype('category').cat.codes)

        # set label column as first column
        cols = list(t.columns)
        cols.insert(0, cols.pop(cols.index('label')))
        t = t.loc[:, cols]
        t = t.drop(['user_id', 'session_id'], axis=1)

        # extract num_rows for each query id: count element in each session
        size = t[['id', 'label']].groupby('id').agg('count')
        c1 = size.index.values
        c2 = size['label'].values
        d = {'id': c1, 'num_rows': c2}
        size_df = pd.DataFrame(d)
        num_rows_df = size_df.drop(['id'], axis=1)

        if save_dataset == True:
            dump_svmlight_file(t.iloc[:, 1:-1].values, t.iloc[:, 0], self.mode + '_' + 'svmlight_' + name + '.txt', query_id=t.id) #TODO mettere i path per bene
            num_rows_df.to_csv( self.mode + '_' + 'svmlight_' + name + '.txt.query', header=None, index=None)

        # X_train = t.iloc[:, 1:-1].values
        # y_train = t.iloc[:, 0].values
        # query_id = t.id.values
        print('Query-based dataset created.')
        # return X_train, y_train, query_id

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
                df[column_name] = pd.Series(encoded_feature, dtype='category')
                continue
            elif column_name in ['user_id', 'session_id']:
                continue
            try:
                df[column_name] = df[column_name].astype(int)
            except:
                print(f'{column_name} contains NaN values')
                df.loc[:, column_name] = df[column_name].fillna(0).astype(int)


        return df, categorical_features
