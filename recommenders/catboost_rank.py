import math
import time
import utils.telegram_bot as HERA
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import data
from recommenders.recommender_base import RecommenderBase
from tqdm import tqdm
from catboost import CatBoost, Pool
from copy import deepcopy
import pickle
import pandas as pd

from utils import check_folder

tqdm.pandas()


class CatboostRanker(RecommenderBase):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.15, iterations=10, max_depth=10, reg_lambda=6.0,
                 colsample_bylevel=1, algo='catboost', one_hot_max_size=255,
                 custom_metric='AverageGain:top=1', include_test=False,
                 file_to_load=None,
                 file_to_store='catboost_2000.sav', limit_trees=False, features_to_one_hot=None):
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

        self.curr_dir = Path(__file__).absolute().parent
        self.data_dir = self.curr_dir.joinpath('..', 'dataset/preprocessed/{}/{}/catboost/'.format(cluster, mode))
        self.target_indices = data.target_indices(mode=mode, cluster=cluster)
        self.features_to_drop = []
        self.include_test = include_test
        self.dataset_name = 'catboost_rank'

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
            'logging_level': 'Verbose',
            'one_hot_max_size': math.ceil(one_hot_max_size),
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'iterations': (2000, 2000),
                                     'max_depth': (11, 11),
                                     'learning_rate': (0.05, 0.05),
                                     'reg_lambda': (7.2, 8.7),
                                     'one_hot_max_size': (511, 511)
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
        train_features = train_df.drop(['user_id', 'session_id', 'item_id', 'label', 'id'], axis=1)

        if self.algo == 'catboost':
            print('Entered')
            features = list(train_features.columns.values)
            self.categorical_features = []
            for f in features:
                if isinstance(train_features.head(1)[f].values[0], str):
                    print(train_features.head(1)[f].values[0])
                    self.categorical_features.append(features.index(f))
                    print(f + ' is categorical!')

            if len(self.categorical_features) == 0:
                self.categorical_features = None


        self.train_features = train_features.columns.values
        X_train = train_features.values
        y_train = train_df['label'].values
        _path = self.data_dir.joinpath('catboost_train.txt.npy')
        queries_train = train_df['id'].values

        print(len(X_train))
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

    def get_scores_batch(self, save=False):
        if self.scores_batch is None:
            self.fit()
            self.recommend_batch()

        base_path = f'dataset/preprocessed/{self.cluster}/{self.mode}/predictions/{self.dataset_name}.pickle'
        check_folder.check_folder(base_path)
        if save:
            with open(base_path, 'wb') as f:
                pickle.dump(self.scores_batch, f)
            print(f'saved at: {base_path}')
        else:
            return self.scores_batch

        return self.scores_batch

    def recommend_batch(self):
        dataset_test = self.get_preprocessed_dataset(mode='test')

        target_indices = data.target_indices(self.mode, self.cluster)
        target_indices.sort()

        test = data.test_df(self.mode, self.cluster)
        print('data for test ready')

        X_test = dataset_test.drop(['user_id', 'session_id', 'item_id', 'label', 'trg_idx'], axis=1).values

        group_id = dataset_test.trg_idx.values

        test_with_weights = Pool(
            data=X_test,
            label=None,
            group_id=group_id,
            cat_features=self.categorical_features
        )

        scores = self.ctb.predict(test_with_weights)

        self.predictions = []
        self.scores_batch = []
        count = 0
        for index in tqdm(target_indices):
            impressions = list(map(int, test.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            scores_impr, sorted_impr = zip(*couples)
            count = count + len(impressions)

            self.predictions.append((index, list(sorted_impr)))
            self.scores_batch.append((index, list(sorted_impr), scores_impr))


        if self.file_to_store is not None:
            self.get_scores_batch(save=True)

        return self.predictions

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

        test_df.drop(['user_id', 'session_id', 'item_id'], inplace=True, axis=1)

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
                    'evaluating recommender {} on {}. Iterations used {}\n MRR is: {}\n\n'.format(self.name,
                                                                                                  self.cluster,
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

        _path = self.data_dir.joinpath('{}.csv'.format(mode))
        classification_df = pd.read_csv(_path)

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
    from utils.menu import mode_selection
    mode = mode_selection()
    model = CatboostRanker(mode=mode, cluster='no_cluster', iterations=10, learning_rate=0.5, algo='catboost')
    model.evaluate(send_MRR_on_telegram=True)


