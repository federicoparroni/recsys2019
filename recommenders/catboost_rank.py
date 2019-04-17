import data
from recommenders.recommender_base import RecommenderBase
from tqdm import tqdm
from catboost import CatBoost, Pool
from copy import deepcopy


tqdm.pandas()

class CatboostRanker(RecommenderBase):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """
    def __init__(self, mode, cluster='no_cluster', iterations=1000, include_test = False,
                 custom_metric='AverageGain:top=1', verbose= False):
        name = 'catboost_rank'
        super(CatboostRanker, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.preds = None
        self.scores_batch = None

        self.test_df = None

        self.target_indices = data.target_indices(mode=mode, cluster=cluster)

        self.iterations = iterations
        self.include_test = include_test
        self.custom_metric = custom_metric
        self.verbose = verbose

        self.default_parameters = {
            'iterations': iterations,
            'custom_metric': custom_metric,
            'verbose': verbose,
            'random_seed': 0
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'learning_rate': (0.01, 0.3),
                                     'max_depth': (2, 6),
                                     'n_estimators': (50, 500),
                                     'reg_lambda': (0, 1),
                                     'reg_alpha': (0, 1)
                                     }


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

        train_df = data.classification_train_df(mode='small', cluster="no_cluster", sparse=False, algo='xgboost')

        #Creating univoque id for each user_id / session_id pair
        train_df = train_df.sort_values(by=['user_id', 'session_id'])
        train_df = train_df.assign(id=(train_df['user_id'] + '_' + train_df['session_id']).astype('category').cat.codes)

        X_train = train_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1).values
        y_train = train_df['label'].values
        queries_train = train_df['id'].values

        #Creating pool train
        train_with_weights = Pool(
            data=X_train,
            label=y_train,
            group_id=queries_train
        )

        test_with_weights = None
        if self.include_test:
            test_df = data.classification_test_df(
                mode=self.mode, sparse=False, cluster=self.cluster)

            print("sorting test")
            test_df = test_df.sort_values(by=['user_id', 'session_id'])

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
        self.ctb = self.fit_model('PairLogit',
                                  additional_params=self.default_parameters,
                                  train_pool=train_with_weights,
                                  test_pool=test_with_weights)
        print('fit done')


    def get_scores_batch(self):
        if self.scores_batch is None:
            self.recommend_batch()
        return self.scores_batch

    def func(self, x):
        #x is a groupd df containing same id (which now is the index)
        target_idx = x.trg_idx.values[0]

        X_test = x.drop(['label', 'trg_idx'], axis=1).values
        y_test = x['label'].values
        queries_test = x.trg_idx.values

        test_with_weights = Pool(
            data=X_test,
            label=None,
            group_id=queries_test
        )

        scores = self.ctb.predict(test_with_weights)

        impr = list(map(int, self.test_df.at[target_idx, 'impressions'].split('|')))

        scores_impr = [[scores[i], impr[i]] for i in range(len(impr))]
        scores_impr.sort(key=lambda x: x[0], reverse=True)

        preds = [x[1] for x in scores_impr]
        scores = [x[0] for x in scores_impr]

        self.predictions.append((target_idx, preds))
        self.scores_batch.append((target_idx, preds, scores))


    def recommend_batch(self):
        test_df = data.classification_test_df(
            mode=self.mode, sparse=False, cluster=self.cluster)

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

        test_df.groupby('trg_idx', as_index=False).progress_apply(self.func)


        return self.predictions



if __name__ == '__main__':
    model = CatboostRanker(mode='small', cluster='no_cluster')
    model.evaluate(send_MRR_on_telegram=False)
