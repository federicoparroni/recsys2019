import lightgbm as lgb
import out
import data
from tqdm import tqdm
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
import numpy as np
import scipy.sparse as sps
from utils.menu import yesno_choice
from utils.check_folder import check_folder
import os
import math
import xgboost as xgb
import data
from recommenders.recommender_base import RecommenderBase
from pandas import Series
from tqdm import tqdm
tqdm.pandas()
from time import time
import matplotlib.pyplot as plt
import utils.telegram_bot as Hera
import datetime
from skopt.space import Real, Integer, Categorical
from utils.reduce_memory_usage_df import reduce_mem_usage
from cython_files.mrr import mrr as mrr_cython
from evaluate.SubEvaluator import SubEvaluator

_x_train = None
_y_train = None
_group_train = None

_x_vali = None
_y_vali = None
_group_test = None



class lightGBM(RecommenderBase):

    def _load_data(self):
        global _x_train, _x_vali, _y_train, _y_vali, _group_train, _group_test, \
                    _user_session_item_train, _user_session_item_test 
        _BASE_PATH = self._BASE_PATH
        if _x_train is None:
            start = time()
            print('Loading data...\n')
            _x_train = pd.read_hdf(f'{_BASE_PATH}/x_train.hdf', key='df').replace(to_replace=-1, value=np.nan)
            _y_train = np.load(f'{_BASE_PATH}/y_train.npy')
            _group_train = np.load(f'{_BASE_PATH}/groups_train.npy')
            #_user_session_item_train = pd.read_csv(f'{_BASE_PATH}/user_session_item_train.csv')
            _x_vali = pd.read_hdf(f'{_BASE_PATH}/x_vali.hdf', key='df').replace(to_replace=-1, value=np.nan)
            _y_vali = np.load(f'{_BASE_PATH}/y_vali.npy')
            _group_test = np.load(f'{_BASE_PATH}/groups_vali.npy')
            print(f'data loaded in: {time() - start}\n')
            #_user_session_item_test = pd.read_csv(f'{_BASE_PATH}/user_session_item_vali.csv')

        self.x_train = _x_train
        self.y_train = _y_train
        self.groups_train = _group_train
        #self.user_session_item_train = _user_session_item_train

        self.x_vali = _x_vali
        self.y_vali = _y_vali
        self.groups_vali = _group_test
        #self.user_session_item_test = _user_session_item_test


    def __init__(self, mode, cluster, dataset_name, params_dict):
        self.dataset_name = dataset_name
        super(lightGBM, self).__init__(
            name=f'lightGBM_{dataset_name}', mode=mode, cluster=cluster)
        self._BASE_PATH = f'dataset/preprocessed/lightGBM/{self.cluster}/{self.mode}/{self.dataset_name}'
        self._load_data()
        self.params_dict = params_dict
        self.eval_res = {}
        self.model = lgb.LGBMRanker(**self.params_dict)

    def fit(self):
        # initialize the model
        self.model.fit(self.x_train, self.y_train, group=self.groups_train, verbose=False)

    def validate(self, min_mrr_to_export=0.668, export_sub=True):
        def _mrr(y_true, y_pred, weight, group):
            l = memoryview(np.array(y_true, dtype=np.int32))
            p = memoryview(np.array(y_pred, dtype=np.float32))
            g = memoryview(np.array(group, dtype=np.int32))
            return 'MRR', mrr_cython(l, p, g,len(g)), True

        def _hera_callback(param):
            iteration_num = param[2]
            if iteration_num % param[1]['print_every'] == 0:
                message = f'PARAMS:\n'
                for k in param[1]:
                    message += f'{k}: {param[1][k]}\n'
                Hera.send_message(f'ITERATION_NUM: {iteration_num}\n {message}\n MRR: {param[5][0][2]}', account='edo')

        # define a callback that will insert whitin the dictionary passed the history of the MRR metric during
        # the training phase
        eval_callback = lgb.record_evaluation(self.eval_res)

        # initialize the model
        self.model.fit(self.x_train, self.y_train, group=self.groups_train, eval_set=[(self.x_vali, self.y_vali)],
                  eval_group=[self.groups_vali], eval_metric=_mrr, eval_names=['validation_set'],
                  early_stopping_rounds=200, verbose=1, callbacks=[eval_callback])

        mrr = self.eval_res['validation_set']['MRR'][self.model.booster_.best_iteration - 1]

        if mrr > min_mrr_to_export:
            # set the path where to save
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
            base_path = f'{self._BASE_PATH}/{time}_{round(mrr,4)}'

            # save the parameters of the model
            check_folder(base_path, point_allowed_path=True)
            print(base_path)
            with open(f"{base_path}/Parameters.txt", "w+") as text_file:
                text_file.write(str(self.params_dict))

            # save the features of the model
            with open(f"{base_path}/used_features.txt", "w+") as text_file:
                text_file.write(str(self.x_train.columns))

            # save the model
            self.model.booster_.save_model(f'{base_path}/{self.name}')

            # save the feature importance of the moodel
            self.plot_features_importance(path=f'{base_path}/feature_importance.png', save=True)

            if export_sub:
                # save the local submission
                recommendations = self.recommend_batch()
                out.create_sub(recommendations, submission_name=self.name, directory=base_path, timestamp_on_name=False)

            #TODO: SAVE ALSO THE SCORES OF THE ALGORITHM

        return mrr

    def plot_features_importance(self, path, save=False):
        plot = lgb.plot_importance(self.model.booster_)
        plt.subplot(plot)
        plt.show()
        if save:
            plt.savefig(path)

    def get_scores_batch(self):
        print('loading target indices')
        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)
        print('done\n')

        full_impressions = data.full_df()

        print('retriving predictions')
        scores = self.model.predict(self.x_vali)
        final_predictions = []
        count = 0
        for index in tqdm(target_indices):
            impressions = list(
                map(int, full_impressions.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            scores_reordered, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr), list(scores_reordered)))
            count = count + len(impressions)
        return final_predictions

    def recommend_batch(self):
        print('loading target indices')
        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)
        print('done\n')

        full_impressions = data.full_df()

        print('retriving predictions')
        scores = self.model.predict(self.x_vali)
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

    @staticmethod
    def get_optimize_params(mode, cluster, dataset_name):
        space = [
            Real(0.01, 0.3, name='learning_rate'),
            Integer(6, 350, name='num_leaves'),
            #Real(0.0, 10, name='reg_lambda'),
            #Real(0.0, 10, name='reg_alpha'),
            Real(0.0, 0.1, name='min_split_gain'),
            Real(0.0, 0.1, name='min_child_weight'),
            Integer(10, 45, name='min_child_samples'),
            #Integer(1, 300, name='min_data_in_leaf'),
            Integer(0, 20, name='bagging_freq'),
            Real(0.6, 1, name='feature_fraction'),
        ]

        def get_mrr(arg_list):

            learning_rate, num_leaves, min_split_gain, min_child_weight, \
                min_child_samples, bagging_freq, feature_fraction = arg_list

            params_dict = {
                'boosting_type': 'gbdt',
                'num_leaves': num_leaves,
                'max_depth': -1,
                'n_estimators': 5000,
                'learning_rate': learning_rate,
                'subsample_for_bin': 200000,
                'class_weights': None,
                #'min_data_in_leaf': min_data_in_leaf,
                'min_split_gain': min_split_gain,
                'min_child_weight': min_child_weight,
                'min_child_samples': min_child_samples,
                'bagging_freq': bagging_freq,
                'feature_fraction': feature_fraction,
                'subsample': 1,
                'subsample_freq': 0,
                'colsample_bytree': 1,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'random_state': None,
                'n_jobs': -1,
                'silent': False,
                'importance_type': 'split',
                'metric': 'None',
                'print_every': 10000,
            }
            lgb=lightGBM(mode=mode, cluster=cluster, dataset_name=dataset_name, params_dict=params_dict)
            mrr = lgb.validate()
            best_it = lgb.model._Booster.best_iteration
            Hera.send_message(f'MRR: {mrr}\n'
                              f'params:\n'
                              f'num_iteration:{best_it}, learning_rate:{learning_rate}, num_leaves:{num_leaves}, '
                              f'min_split_gain: {min_split_gain}, min_child_weight: {min_child_weight}, min_child_samples: {min_child_samples}')
            return -mrr
        return space, get_mrr

    def fit_cv(self, x, y, groups, train_indices, test_indices, **fit_params):
        X_train = x.reset_index(drop=True).loc[train_indices]
        y_train = y[train_indices]
        _, group = np.unique(groups[train_indices], return_counts=True)
        self.model.fit(X_train, y_train, group=group)

    def get_scores_cv(self, x, groups, test_indices):
        # check if x is the dataset for train or test
        if x.shape[0] == len(test_indices):
            user_session_item = self.user_session_item_test
        else:
            user_session_item = self.user_session_item_train
        # filter by index
        X_test = x.reset_index(drop=True).loc[test_indices]
        preds = list(self.model.predict(X_test))
        # add scores to dataset
        user_session_item = user_session_item.reset_index(drop=True).loc[test_indices]
        user_session_item['score_lightgbm'] = preds
        return user_session_item

if __name__ == '__main__':
    params_dict = {
        'boosting_type':'gbdt',
        'num_leaves': 64,
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample_for_bin': 200000,
        'class_weights': None,
        'min_split_gain': 0.0,
        'min_child_weight': 0.0,
        'min_child_samples': 20,
        #'min_data_in_leaf':1,
        'subsample':1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'random_state': None,
        'n_jobs': -1,
        'silent': False,
        'importance_type': 'split',
        'metric': 'None',
        'print_every': 1000,
        'first_only': True
    }
    dataset_name = input('dataset name: \n')
    model = lightGBM(mode='local', cluster='no_cluster', dataset_name=dataset_name, params_dict=params_dict)
    model.validate()
    #model.plot_features_importance()





