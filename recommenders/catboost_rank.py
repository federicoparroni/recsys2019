import math
import time
import utils.telegram_bot as HERA

import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.15, iterations=10, max_depth=10, reg_lambda=2.0,
                 colsample_bylevel=1, bootstrap_type='Bernoulli', subsample = 0.66,
                 custom_metric='AverageGain:top=1', include_test=False,
                 file_to_load=None,
                 file_to_store=None, limit_trees=False, features_to_one_hot=None):
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
            'learning_rate': learning_rate,
            'max_depth': math.ceil(max_depth),
            'colsample_bylevel': math.ceil(colsample_bylevel),
            'reg_lambda': reg_lambda,
            'leaf_estimation_iterations': 25,
            'leaf_estimation_method': 'Newton',
            'boosting_type': 'Plain',
            'loss_function': 'QuerySoftMax',
            'train_dir': 'QuerySoftMax',
            'bootstrap_type': bootstrap_type,
            'logging_level': 'Verbose',
            'subsample': subsample
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'iterations': (50, 50),
                                     'max_depth': (10.5, 11),
                                     'learning_rate': (0.15, 0.15),
                                     'reg_lambda': (6.5, 6.75),
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

        self.ctb = None
        self.categorical_features = None
        self.train_features = None
        self.scores_batch = None

    def fit_model(self, additional_params=None, train_pool=None, test_pool=None):
        parameters = deepcopy(self.default_parameters)

        if additional_params is not None:
            parameters.update(additional_params)

        model = CatBoost(parameters)
        start = time.time()

        model.fit(train_pool, eval_set=test_pool, plot=False)

        end = time.time()
        print('With {} iteration, training took {} sec'.format(parameters['iterations'], end - start))

        return model

    def get_feature_importance(self, additional_params=None, train_pool=None):
        parameters = deepcopy(self.default_parameters)

        if additional_params is not None:
            parameters.update(additional_params)

        model = CatBoost(parameters)
        features_imp = model.get_feature_importance(train_pool)

        return features_imp

    def fit(self, is_train=True):

        if self.file_to_load is not None:
            # --- To load model ---
            self.ctb = pickle.load(open(self.file_to_load, 'rb'))
            print("Model loaded")
            return

        train_df = self.get_preprocessed_dataset(mode='train')

        train_features = train_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1)

        self.train_features = train_features.columns.values

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

        if is_train:
            self.ctb = self.fit_model(train_pool=train_with_weights,
                                      test_pool=test_with_weights)
        else:
            self.ctb = self.get_feature_importance(train_pool=train_with_weights)
        print('fit done')

        # ----To store model----
        if self.file_to_store is not None:
            pickle.dump(self.ctb, open(self.file_to_store, 'wb'))  # pickling

    def get_scores_batch(self):
        if self.scores_batch is None:
            self.fit()
            self.recommend_batch()
        return self.scores_batch[1:]

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

        if self.limit_trees and self.limit_trees > 0:
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

        test_df = self.get_preprocessed_dataset(mode='test')

        test_df.drop(['user_id', 'session_id'], inplace=True, axis=1)

        self.predictions = []
        self.scores_batch = []

        test_df.groupby('trg_idx', as_index=False).progress_apply(self.func)

        return self.predictions[1:]

    def set_limit_trees(self, n):
        if n > 0:
            self.limit_trees = n

    def get_feature_importance_results(self):
        if self.ctb is None:
            print('Training ctb first')
            self.fit(is_train=False)

        feat_imp = list(self.ctb)
        tuples_features = list(zip(self.train_features, feat_imp))
        sorted(tuples_features, key=lambda x: x[1])

        features = [el[0] for el in list(zip(*tuples_features))]
        score = [el[1] for el in list(zip(*tuples_features))]

        x_pos = np.arange(len(features))

        plt.bar(x_pos, score, align='center')
        plt.xticks(x_pos, features)
        plt.ylabel('Popularity Score')
        plt.show()

    def iterations_validation(self, max_trees, range_step=25, mode='auto'):
        if self.ctb is None:
            self.fit()

        test_df = self.get_preprocessed_dataset(mode='test')

        test_df.drop(['user_id', 'session_id'], inplace=True, axis=1)

        if mode == 'auto':
            list_num_trees = [max_trees - i * range_step for i in range(max_trees)]

            for trees in list_num_trees:
                self.set_limit_trees(trees)

                self.predictions = []
                self.scores_batch = []
                test_df.groupby('trg_idx', as_index=False).progress_apply(self.func)

                MRR = self.compute_MRR(self.predictions[1:])
                HERA.send_message(
                    'evaluating recommender {} on {}. Iterations used {}\n MRR is: {}\n\n'.format(self.name,
                                                                                                  self.cluster, trees,
                                                                                                  MRR))
        else:
            while True:
                # Getting user input
                while True:
                    trees = input("How many iterations?")
                    try:
                        self.set_limit_trees(int(trees))
                        break
                    except ValueError:
                        pass

                self.predictions = []
                self.scores_batch = []
                test_df.groupby('trg_idx', as_index=False).progress_apply(self.func)

                MRR = self.compute_MRR(self.predictions[1:])
                HERA.send_message(
                    'evaluating recommender {} on {}. Iterations used {}\n MRR is: {}\n\n'.format(self.name, self.cluster,
                                                                                                  trees, MRR))

    def get_preprocessed_dataset(self, mode):
        """
        Apply preprocessing steps to dataset
        - add id to identify groups
        - keep only useful columns
        - join eventual one-hotted categorical features
        :param mode:
        :return:
        """

        if mode == 'test':
            classification_df = data.classification_test_df(
                mode=self.mode, sparse=False, cluster=self.cluster, algo=self.algo).copy()
        else:
            classification_df = data.classification_train_df(
                mode=self.mode, sparse=False, cluster=self.cluster, algo=self.algo).copy()

        classification_df = classification_df.reindex(
            'user_id,session_id,label,times_impression_appeared,time_elapsed_from_last_time_impression_appeared,impression_position,steps_from_last_time_impression_appeared,price,price_position,impression_position_wrt_last_interaction,impression_position_wrt_second_last_interaction,clickout_item_session_ref_this_impr,interaction_item_deals_session_ref_this_impr,interaction_item_image_session_ref_this_impr,interaction_item_info_session_ref_this_impr,interaction_item_rating_session_ref_this_impr,search_for_item_session_ref_this_impr,clickout_item_session_ref_not_in_impr,interaction_item_deals_session_ref_not_in_impr,interaction_item_image_session_ref_not_in_impr,interaction_item_info_session_ref_not_in_impr,interaction_item_rating_session_ref_not_in_impr,search_for_item_session_ref_not_in_impr,session_length_in_step,session_length_in_time,time_per_step,frenzy_factor,average_price_position,avg_price_interacted_item,avg_pos_interacted_items_in_impressions,pos_last_interaction_in_impressions,search_for_poi_distance_from_last_clickout,search_for_poi_distance_from_first_action,change_sort_order_distance_from_last_clickout,change_sort_order_distance_from_first_action,time_passed_before_clk'.split(','), axis=1)

        if len(self.features_to_drop) > 0:
            classification_df.drop(self.features_to_drop, axis=1, inplace=True)

        print('Lenght is {}, features are {}'.format(classification_df.shape[0], classification_df.shape[1]))

        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)

        if mode == 'test':
            self.test_df = data.test_df(self.mode, self.cluster)

            sessi_target = self.test_df.loc[target_indices].session_id.values
            self.dict_session_trg_idx = dict(zip(sessi_target, target_indices))

            classification_df['trg_idx'] = classification_df.apply(
                lambda row: self.dict_session_trg_idx.get(row.session_id), axis=1)

        else:
            # Creating univoque id for each user_id / session_id pair
            classification_df = classification_df.sort_values(by=['user_id', 'session_id'])
            classification_df = classification_df.assign(
                id=(classification_df['user_id'] + '_' + classification_df['session_id']).astype('category').cat.codes)

        if self.features_to_one_hot is not None:
            for f in self.features_to_one_hot:
                one_hot = pd.get_dummies(classification_df[f])
                classification_df = classification_df.drop([f], axis=1)
                classification_df = classification_df.join(one_hot)

        return classification_df


if __name__ == '__main__':
    model = CatboostRanker(mode='local', cluster='no_cluster', file_to_load='cat_1200')
    array = model.get_scores_batch()
    np.save('localscores_catboost.npy', array)
    model = CatboostRanker(mode='full', cluster='no_cluster', file_to_load='cat_1200')
    array = model.get_scores_batch()
    np.save('fullscores_catboost.npy', array)