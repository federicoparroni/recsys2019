import math
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
import os
from utils.check_folder import check_folder
from random_validator import RandomValidator

tqdm.pandas()


class CatboostRanker(RecommenderBase):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.25, iterations=150, max_depth=12, reg_lambda=13.465,
                 colsample_bylevel=1, algo='catboost', one_hot_max_size=42, leaf_estimation_iterations=25,
                 custom_metric='AverageGain:top=1', include_test=True,
                 file_to_load=None, loss_function='YetiRank', train_dir='YetiRank',
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

        self.curr_dir = Path(__file__).absolute().parent
        self.data_dir = self.curr_dir.joinpath('..', 'dataset/preprocessed/{}/{}/catboost/'.format(cluster, mode))
        self.target_indices = data.target_indices(mode=mode, cluster=cluster)
        self.features_to_drop = []

        self.include_test = include_test
        if self.mode=='full':
            self.include_test = False
        self.dataset_name = 'catboost_rank'

        self.default_parameters = {
            'iterations': math.ceil(iterations),
            'custom_metric': custom_metric,
            'learning_rate': learning_rate,
            'max_depth': math.ceil(max_depth),
            'colsample_bylevel': math.ceil(colsample_bylevel),
            'reg_lambda': reg_lambda,
            'leaf_estimation_iterations': math.ceil(leaf_estimation_iterations),
            'leaf_estimation_method': 'Newton',
            'boosting_type': 'Plain',
            'loss_function': loss_function,
            'train_dir': train_dir,
            'logging_level': 'Verbose',
            'one_hot_max_size': math.ceil(one_hot_max_size),
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'iterations': (50, 50),
                                     'max_depth': (12, 12),
                                     'learning_rate': (0.25, 0.25),
                                     'reg_lambda': (13.465, 13.465),
                                     'one_hot_max_size': (42, 42),
                                     'leaf_estimation_iterations': [20, 22, 24, 25, 27, 29, 31, 33, 35]
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
        model.fit(train_pool, eval_set=test_pool, plot=False)
        return model

    def fit_cv(self, x, y, groups, train_indices, test_indices, **fit_params):
        parameters = deepcopy(self.default_parameters)

        if fit_params is not None:
            parameters.update(fit_params)

        self.ctb = CatBoost(parameters)

        train_values = x.drop(['label', 'id'], axis=1)

        features = list(train_values.columns.values)
        self.categorical_features = []
        for f in features:
            if isinstance(train_values.head(1)[f].values[0], str) or f == 'day' or f == 'past_closest_action_involving_impression' or f =='future_closest_action_involving_impression':
                self.categorical_features.append(features.index(f))
                print(f + ' is categorical!')

        if len(self.categorical_features) == 0:
            self.categorical_features = None

        train_with_weights = Pool(
            data=train_values.values[train_indices, :],
            label=x['label'].values[train_indices],
            group_id=x['id'].values[train_indices],
            cat_features=self.categorical_features
        )

        self.ctb.fit(train_with_weights, plot=False)

        return self.ctb

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
            _path = 'dataset/preprocessed/{}/{}/catboost/model/{}'.format(self.cluster, self.mode, self.file_to_load)
            check_folder(_path)
            self.ctb = pickle.load(open(_path, 'rb'))
            print("Model loaded")
            return

        train_df = self.get_preprocessed_dataset(mode='train')
        train_features = train_df.drop(['user_id', 'session_id', 'item_id', 'label', 'id'], axis=1)

        if self.algo == 'catboost':
            features = list(train_features.columns.values)
            self.categorical_features = []
            for f in features:
                if isinstance(train_features.head(1)[f].values[0], str) or f == 'day' or f == 'past_closest_action_involving_impression' or f =='future_closest_action_involving_impression':
                    self.categorical_features.append(features.index(f))
                    print(f + ' is categorical!')

            if len(self.categorical_features) == 0:
                self.categorical_features = None


        self.train_features = train_features.columns.values
        X_train = train_features.values
        y_train = train_df['label'].values
        _path = self.data_dir.joinpath('catboost_train.txt.npy')
        queries_train = train_df['id'].values

        # Creating pool for training data
        train_with_weights = Pool(
            data=X_train,
            label=y_train,
            group_id=queries_train,
            cat_features=self.categorical_features
        )

        test_with_weights = None

        if self.include_test:
            dataset_test = self.get_preprocessed_dataset(mode='test')
            target_indices = data.target_indices(self.mode, self.cluster)
            target_indices.sort()

            print('data for test ready')

            test_feat_df = dataset_test.drop(['user_id', 'session_id', 'item_id', 'label', 'id'], axis=1)

            if list(test_feat_df.columns.values) != list(self.train_features):
                print('Training columns are different from test columns! Check')
                print(self.train_features)
                print(test_feat_df.columns.values)
                exit(0)

            X_test = test_feat_df.values

            group_id = dataset_test.id.values

            test_with_weights = Pool(
                data=X_test,
                label=dataset_test.label.values,
                group_id=group_id,
                cat_features=self.categorical_features
            )
            self.test_with_weights = test_with_weights


        print('data for train ready')

        if is_train:
            self.ctb = self.fit_model(train_pool=train_with_weights,
                                      test_pool=test_with_weights)
        else:
            self.ctb = self.get_feature_importance(train_pool=train_with_weights)

        print('fit done')

        # ----To store model----
        if self.file_to_store is not None:
            _path = 'dataset/preprocessed/{}/{}/catboost/model/{}'.format(self.cluster, self.mode, self.file_to_store)
            check_folder(_path)
            pickle.dump(self.ctb, open(_path, 'wb'))


    def get_scores_batch(self):
        if self.ctb is None:
            self.fit()

        if self.scores_batch is None:
            self.recommend_batch()

        return self.scores_batch


    def recommend_batch(self):
        test = data.test_df(self.mode, self.cluster)
        target_indices = data.target_indices(self.mode, self.cluster)
        target_indices.sort()

        if self.include_test:
            test_with_weights = self.test_with_weights
        else:
            dataset_test = self.get_preprocessed_dataset(mode='test')

            print('data for test ready')

            test_feat_df = dataset_test.drop(['user_id', 'session_id', 'item_id', 'label', 'id'], axis=1)

            if list(test_feat_df.columns.values) != list(self.train_features):
                print('Training columns are different from test columns! Check')
                print(self.train_features)
                print(test_feat_df.columns.values)
                exit(0)

            X_test = test_feat_df.values

            group_id = dataset_test.id.values

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
            if len(couples)>0:
                scores_impr, sorted_impr = zip(*couples)
                count = count + len(impressions)

                self.predictions.append((index, list(sorted_impr)))
                self.scores_batch.append((index, list(sorted_impr), scores_impr))


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
                test_df.groupby('id', as_index=False).progress_apply(self.func)

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
                test_df.groupby('id', as_index=False).progress_apply(self.func)

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

        if mode == 'train':
            classification_df = data.dataset_catboost_train(self.mode, self.cluster).copy()
        elif mode == 'test':
            classification_df = data.dataset_catboost_test(self.mode, self.cluster).copy()
        else:
            print('Wrong mode given in get_preprocessed_dataset!')
            return

        if len(self.features_to_drop) > 0:
            classification_df.drop(self.features_to_drop, axis=1, inplace=True)

        print('Lenght of dataset {} is {}, features numbers are {}'.format(mode, classification_df.shape[0], classification_df.shape[1]))

        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)

        if mode == 'test':
            self.test_df = data.test_df(self.mode, self.cluster)

            sessi_target = self.test_df.loc[target_indices].session_id.values
            self.dict_session_id = dict(zip(sessi_target, target_indices))

            classification_df['id'] = classification_df.apply(
                lambda row: self.dict_session_id.get(row.session_id), axis=1)

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

    def get_scores_cv(self, x, groups, test_indices):

        test_with_weights = Pool(
            data=x.drop(['id', 'label'], axis=1).values[test_indices, :],
            label=None,
            group_id=x.id.values[test_indices],
            cat_features=self.categorical_features
        )
        preds = list(self.ctb.predict(test_with_weights))
        test_indices = list(test_indices)

        if x.shape[0] == len(test_indices):
            user_session_item = data.dataset_catboost_test(mode=self.mode, cluster=self.cluster).copy()
        else:
            user_session_item = data.dataset_catboost_train(mode=self.mode, cluster=self.cluster).copy()

        user_session_item = user_session_item[['user_id', 'session_id', 'item_id', 'index']]
        user_session_item = user_session_item.loc[test_indices]
        print('Len of resulting df is {} \nLen of test indices list is {}\nLEN OF PREDS: {}'.format(len(user_session_item), len(test_indices), len(preds)))
        import time
        time.sleep(1)
        print(test_indices[:50])
        print(list(user_session_item.index)[:50])
        print(preds[:50])
        user_session_item['score_catboost'] = preds

        return user_session_item


if __name__ == '__main__':
    from utils.menu import mode_selection, options, cluster_selection

    mode = mode_selection()
    cluster = cluster_selection()
    model = CatboostRanker(mode=mode, cluster=cluster, iterations=150, learning_rate=0.25, algo='catboost')

    sel = options(['evaluate', 'export the sub', 'export the scores'], ['evaluate', 'export the sub',
                                                                        'export the scores'],
                  'what do you want to do after model fitting and the recommendations?')

    import time
    time.sleep(0)
    if 'export the sub' in sel and 'export the scores' in sel:
        model.run(export_sub=True, export_scores=True)
    elif 'export the sub' in sel and 'export the scores' not in sel:
        model.run(export_sub=True, export_scores=False)
    elif 'export the sub' not in sel and 'export the scores' in sel:
        model.run(export_sub=False, export_scores=True)

    if 'evaluate' in sel and ('export the sub' in sel or 'export the scores' in sel):
        model.evaluate(send_MRR_on_telegram=True, already_fitted=True)
    elif 'evaluate' in sel:
        model.evaluate(send_MRR_on_telegram=True, already_fitted=False)

    #model.evaluate(send_MRR_on_telegram=True)
    r = RandomValidator(model, automatic_export=False)
    r.validate(100)


