
import data
from recommenders.recommender_base import RecommenderBase
from tqdm import tqdm
from catboost import CatBoost, Pool
from copy import deepcopy
import pickle
tqdm.pandas()

import os
os.chdir("../")
print(os.getcwd())

class CatboostRanker(RecommenderBase):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """


    def __init__(self, mode, cluster='no_cluster', learning_rate=0.05, iterations=200, max_depth = 8, reg_lambda=3, colsample_bylevel = 1,
                 custom_metric='AverageGain:top=1', verbose= False, include_test = False, file_to_load = None, file_to_store = None, limit_trees = False):
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
        self.iterations = iterations
        self.include_test = include_test
        self.custom_metric = custom_metric
        self.verbose = verbose

        self.default_parameters = {
            'iterations': iterations,
            'custom_metric': custom_metric,
            'verbose': verbose,
            'random_seed': 0,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'colsample_bylevel': colsample_bylevel,
            'reg_lambda': reg_lambda
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'iterations': (10, 1000),
                                     'max_depth': (2, 8),
                                     'learning_rate': (0.01, 0.2),
                                     'reg_lambda': (0.1, 5),
                                     'colsample_bylevel': (0.5, 1)
                                     }


        self.limit_trees = False
        self.file_to_load = 'cat_no_feat.sav'
        self.file_to_store = None

    def fit_model(self, loss_function, additional_params=None, train_pool=None, test_pool=None):
        parameters = deepcopy(self.default_parameters)
        parameters['loss_function'] = loss_function
        parameters['train_dir'] = loss_function

        if additional_params is not None:
            parameters.update(additional_params)

        model = CatBoost(parameters)
        model.fit(train_pool, eval_set=test_pool, plot=True)

        return model

    def fit(self):

        if self.file_to_load is not None:
            self.ctb = pickle.load(open(self.file_to_load, 'rb'))
            print("File loaded")
            return

        train_df = data.classification_train_df(mode=self.mode, cluster=self.cluster, sparse=False, algo='xgboost')
        #
        # prop = pd.read_csv("dataset/preprocessed/no_cluster/small/feature/frenzy_factor_session/features.csv".format(self.cluster, self.mode))
        # train_df = pd.merge(train_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])
        #
        # prop = pd.read_csv("dataset/preprocessed/no_cluster/small/feature/mean_cheap_price_position_clickout/features.csv".format(self.cluster, self.mode))
        # train_df = pd.merge(train_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])
        #
        # prop = pd.read_csv("dataset/preprocessed/no_cluster/small/feature/time_passed_before_clk/features.csv".format(self.cluster, self.mode))
        # train_df = pd.merge(train_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])
        #
        # prop = pd.read_csv("dataset/preprocessed/no_cluster/small/feature/mean_interacted_position/features.csv".format(self.cluster, self.mode))
        # train_df = pd.merge(train_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])

        #Creating univoque id for each user_id / session_id pair
        train_df = train_df.sort_values(by=['user_id', 'session_id'])
        train_df = train_df.assign(id=(train_df['user_id'] + '_' + train_df['session_id']).astype('category').cat.codes)

        X_train = train_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1).values
        y_train = train_df['label'].values
        queries_train = train_df['id'].values

        #Creating pool for training data
        train_with_weights = Pool(
            data=X_train,
            label=y_train,
            group_id=queries_train
        )

        test_with_weights = None
        if self.include_test:
            test_df = data.classification_test_df(
                mode=self.mode, sparse=False, cluster=self.cluster)

            test_df = test_df.sort_values(by=['user_id', 'session_id'])

            if 'Unnamed: 0.1' in test_df.columns.values:
                test_df = test_df.set_index(['Unnamed: 0.1'])

            test_df['id'] = test_df.groupby(['user_id', 'session_id']).ngroup()

            X_test = test_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1).values
            y_test = test_df['label'].values
            queries_test = test_df['id'].values

            print("pooling")
            test_with_weights = Pool(
                data=X_test,
                label=y_test,
                group_id=queries_test
            )

        print('data for train ready')
        self.ctb = self.fit_model('QuerySoftMax', self.default_parameters,
                                  train_pool=train_with_weights,
                                  test_pool=test_with_weights)
        print('fit done')

        # ----To store model----------
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

        X_test = x.drop(['label', 'trg_idx'], axis=1).values
        # useless
        # y_test = x['label'].values
        group_id = x.trg_idx.values

        test_with_weights = Pool(
            data=X_test,
            label=None,
            group_id=group_id
        )

        if self.limit_trees is not None:
            print('Limiting trees used in the algo to {} trees'.format(self.limit_trees))
            scores = self.ctb.predict(test_with_weights, ntree_end=self.limit_trees)
        else:
            scores = self.ctb.predict(test_with_weights)

        impr = list(map(int, self.test_df.at[target_idx, 'impressions'].split('|')))

        min_len = len(scores)
        if len(scores) != len(impr):
            print("At session" + self.test_df.at[target_idx, 'session_id'])
            print(x.impression_position)
            print(impr)
            print(scores)
        if len(scores) > len(impr):
            min_len = len(impr)

        scores_impr = [[scores[i], impr[i]] for i in range(min_len)]

        #Order by max score
        scores_impr.sort(key=lambda x: x[0], reverse=True)

        preds = [x[1] for x in scores_impr]
        scores = [x[0] for x in scores_impr]

        self.predictions.append((target_idx, preds))
        self.scores_batch.append((target_idx, preds, scores))


    def recommend_batch(self):

        test_df = data.classification_test_df(
            mode=self.mode, sparse=False, cluster=self.cluster)

        if 'Unnamed: 0.1' in test_df.columns.values:
            test_df = test_df.set_index(['Unnamed: 0.1'])

        print("Merging")

        # prop = pd.read_csv("dataset/preprocessed/{}/{}/feature/frenzy_factor_session/features.csv".format(self.cluster, self.mode))
        # test_df = pd.merge(test_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])
        #
        # prop = pd.read_csv("dataset/preprocessed/{}/{}/feature/mean_cheap_price_position_clickout/features.csv".format(self.cluster, self.mode))
        # test_df = pd.merge(test_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])
        #
        # prop = pd.read_csv("dataset/preprocessed/{}/{}/feature/time_passed_before_clk/features.csv".format(self.cluster, self.mode))
        # test_df = pd.merge(test_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])
        #
        # prop = pd.read_csv("dataset/preprocessed/{}/{}/feature/time_passed_before_clk/features.csv".format(self.cluster, self.mode))
        # test_df = pd.merge(test_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])
        #
        # prop = pd.read_csv("dataset/preprocessed/{}/{}/feature/mean_interacted_position/features.csv".format(self.cluster, self.mode))
        # test_df = pd.merge(test_df, prop, left_on=['user_id', 'session_id'], right_on=['user_id', 'session_id'])

        print(test_df.shape[0])
        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)

        print("loading test df")
        self.test_df = data.test_df(self.mode, self.cluster)

        sessi_target = self.test_df.loc[target_indices].session_id.values
        self.dict_session_trg_idx = dict(zip(sessi_target, target_indices))

        print("applying trg_idx")
        test_df['trg_idx'] = test_df.apply(lambda row: self.dict_session_trg_idx.get(row.session_id), axis=1)

        print("dropping")
        test_df.drop(['user_id', 'session_id'], inplace=True, axis=1)
        test_df = test_df.sort_values(by=['trg_idx', 'impression_position'])

        self.predictions = []
        self.scores_batch = []

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
    model = CatboostRanker(mode='full', cluster='no_cluster', iterations=500, include_test=False)
    model.run()
    #model.evaluate(send_MRR_on_telegram=False)
