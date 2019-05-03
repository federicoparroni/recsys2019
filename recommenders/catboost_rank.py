import math

import data
from recommenders.recommender_base import RecommenderBase
from tqdm import tqdm
from catboost import CatBoost, Pool
from copy import deepcopy
import pickle
import pandas as pd
tqdm.pandas()

class CatboostRanker(RecommenderBase):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.15, iterations=200, max_depth=8, reg_lambda=3,
                 colsample_bylevel=1,
                 custom_metric='AverageGain:top=1', algo='xgboost', verbose=False, include_test=False, file_to_load=None,
                 file_to_store=None, limit_trees=False, features_to_one_hot = None):
        """
        :param mode:
        :param cluster:
        :param iterations: number of trees to use
        :param include_test: True if use test for allow visual early stopping
        :param custom_metric: Metric to evaluate during training
        :param verbose: True if writing a log of training
        :param file_to_load: specify the path of an existing model to use without training a new one
        :param file_to_store: specify the path where the model will be stored
        :param limit_trees: limit trees to use whenever an existing model is being used
        """
        name = 'catboost_rank'
        super(CatboostRanker, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.target_indices = data.target_indices(mode=mode, cluster=cluster)
        self.features_to_drop = []
        self.include_test = include_test

        self.default_parameters = {
            'iterations': math.ceil(iterations),
            'custom_metric': custom_metric,
            'verbose': verbose,
            'random_seed': 0,
            'learning_rate': learning_rate,
            'max_depth': math.ceil(max_depth),
            'colsample_bylevel': math.ceil(colsample_bylevel),
            'reg_lambda': math.ceil(reg_lambda),
            'loss_function': 'QuerySoftMax',
            'train_dir': 'QuerySoftMax'
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'iterations': (10, 50),
                                     'max_depth': (3, 8),
                                     'learning_rate': (0.01, 0.2),
                                     'reg_lambda': (1, 5),
                                     }

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'colsample_bylevel': 1
        }

        self.limit_trees = limit_trees
        self.file_to_load = file_to_load
        self.file_to_store = file_to_store
        self.features_to_one_hot = features_to_one_hot
        self.algo = algo

        self.categorical_features = None

    def fit_model(self, additional_params=None, train_pool=None, test_pool=None):
        parameters = deepcopy(self.default_parameters)

        if additional_params is not None:
            parameters.update(additional_params)

        model = CatBoost(parameters)
        model.fit(train_pool, eval_set=test_pool, plot=True)

        return model

    def fit(self):

        if self.file_to_load is not None:
            # --- To load model ---
            self.ctb = pickle.load(open(self.file_to_load, 'rb'))
            print("Model loaded")
            return

        print('Start training the model...')
        train_df = data.classification_train_df(mode=self.mode, cluster=self.cluster, sparse=False, algo=self.algo)

        print('Shape of train is ' + str(train_df.shape[0] ))

        if train_df.shape[0] > 10000000:
            print('keeping first 100000...')
            train_df = train_df[:10000000]

        if 'times_doubleclickout_on_item' in train_df.columns.values:
            train_df = train_df.drop(['times_doubleclickout_on_item'], axis=1)

        if len(self.features_to_drop)>0:
            train_df.drop(self.features_to_drop, axis=1, inplace=True)

        train_df = train_df.drop(['times_doubleclickout_on_item'], axis=1)

        #train_df.drop(['avg_price_interacted_item','average_price_position', 'avg_pos_interacted_items_in_impressions', 'pos_last_interaction_in_impressions'], axis=1, inplace=True)
        print(train_df.shape[1])

        if self.features_to_one_hot is not None:
            for f in self.features_to_one_hot:
                one_hot = pd.get_dummies(train_df[f])
                train_df = train_df.drop([f], axis=1)
                train_df = train_df.join(one_hot)

        # Creating univoque id for each user_id / session_id pair
        train_df = train_df.sort_values(by=['user_id', 'session_id'])
        train_df = train_df.assign(id=(train_df['user_id'] + '_' + train_df['session_id']).astype('category').cat.codes)

        train_features = train_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1)

        X_train = train_features.values
        y_train = train_df['label'].values
        queries_train = train_df['id'].values

        if self.algo == 'catboost':
            features = list(train_features.columns.values)
            self.categorical_features = []
            for f in features:
                if isinstance(train_features.head(1)[f].values[0], str):
                    self.categorical_features.append(features.index(f))
                    print(f + ' is categorical!')

            if len(self.categorical_features) == 0:
                self.categorical_features = None

        # Creating pool for training data
        train_with_weights = Pool(
            data=X_train,
            label=y_train,
            group_id=queries_train,
            cat_features=self.categorical_features
        )

        test_with_weights = None

        if self.include_test:
            test_df = data.classification_test_df(
                mode=self.mode, sparse=False, cluster=self.cluster)

            test_df = test_df.sort_values(by=['user_id', 'session_id'])

            test_df['id'] = test_df.groupby(['user_id', 'session_id']).ngroup()

            X_test = test_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1).values
            y_test = test_df['label'].values
            queries_test = test_df['id'].values

            print("pooling")
            test_with_weights = Pool(
                data=X_test,
                label=y_test,
                group_id=queries_test,
                cat_features=self.categorical_features
            )

        print('data for train ready')
        self.ctb = self.fit_model(self.default_parameters,
                                  train_pool=train_with_weights,
                                  test_pool=test_with_weights)
        print('fit done')

        # ----To store model----
        if self.file_to_store is not None:
            pickle.dump(self.ctb, open(self.file_to_store, 'wb'))  # pickling

    def get_scores_batch(self):
        if self.scores_batch is None:
            self.recommend_batch()
        return self.scores_batch

    def func(self, x):
        """
        Func given to progress_apply to create recommendations given a dataset for catboost
        :param x: groupd df containing same trg_idx (which is the index to return in the tuples)
        :return: tuple (trg_idx, list of recs)
        """

        target_idx = x.trg_idx.values[0]

        x = x.sort_values(by=['impression_position'])

        X_test = x.drop(['label', 'trg_idx'], axis=1).values

        # useless
        # y_test = x['label'].values
        group_id = x.trg_idx.values

        test_with_weights = Pool(
            data=X_test,
            label=None,
            group_id=group_id,
            cat_features=self.categorical_features
        )

        if self.limit_trees and self.limit_trees>0:
            scores = self.ctb.predict(test_with_weights, ntree_end=self.limit_trees)
        else:
            scores = self.ctb.predict(test_with_weights)

        impr = list(map(int, self.test_df.at[target_idx, 'impressions'].split('|')))

        min_len = len(scores)
        if len(scores) != len(impr):
            print("At session" + self.test_df.at[target_idx, 'session_id'] + 'found different len of scores wrt len '
                                                                             'of impressions')
            print(x.impression_position)
            print(impr)
            print(scores)
        if len(scores) > len(impr):
            min_len = len(impr)

        scores_impr = [[scores[i], impr[i]] for i in range(min_len)]

        # Order by max score
        scores_impr.sort(key=lambda x: x[0], reverse=True)

        preds = [x[1] for x in scores_impr]
        scores = [x[0] for x in scores_impr]

        self.predictions.append((target_idx, preds))
        self.scores_batch.append((target_idx, preds, scores))

    def recommend_batch(self):

        test_df = data.classification_test_df(
            mode=self.mode, sparse=False, cluster=self.cluster, algo=self.algo).copy()

        test_df = test_df.sort_values(by=['user_id', 'session_id', 'impression_position'])

        #test_df.drop(['avg_price_interacted_item','average_price_position', 'avg_pos_interacted_items_in_impressions', 'pos_last_interaction_in_impressions'], axis=1, inplace=True)
        if len(self.features_to_drop) > 0:
            test_df.drop(self.features_to_drop, axis=1, inplace=True)

        if 'Unnamed: 0' in test_df.columns.values:
            test_df = test_df.drop(['Unnamed: 0'], axis=1)

        if 'times_doubleclickout_on_item' in test_df.columns.values:
            test_df = test_df.drop(['times_doubleclickout_on_item'], axis=1)
        print(test_df.shape[0])
        print(test_df.shape[1])

        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)

        self.test_df = data.test_df(self.mode, self.cluster)

        sessi_target = self.test_df.loc[target_indices].session_id.values
        self.dict_session_trg_idx = dict(zip(sessi_target, target_indices))

        test_df['trg_idx'] = test_df.apply(lambda row: self.dict_session_trg_idx.get(row.session_id), axis=1)

        test_df.drop(['user_id', 'session_id'], inplace=True, axis=1)

        self.predictions = []
        self.scores_batch = []

        if self.features_to_one_hot is not None:
            for f in self.features_to_one_hot:
                one_hot = pd.get_dummies(test_df[f])
                test_df = test_df.drop([f], axis=1)
                test_df = test_df.join(one_hot)

        # while True:
        #     timeNum = input("How many iterations?")
        #     try:
        #         self.set_limit_trees(int(timeNum))
        #         break
        #     except ValueError:
        #         pass

        test_df.groupby('trg_idx', as_index=False).progress_apply(self.func)

        return self.predictions

    def set_limit_trees(self, n):
        if n > 0:
            self.limit_trees = n

if __name__ == '__main__':
    model = CatboostRanker(mode='small', cluster='no_cluster', iterations=10, include_test=False, algo='xgboost')
    model.evaluate(send_MRR_on_telegram=False)
