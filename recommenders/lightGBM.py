import lightgbm as lgb
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

_x_train = None
_y_train = None
_group_train = None

_x_vali = None
_y_vali = None
_group_test = None



class lightGBM(RecommenderBase):

    def _load_data(self):
        global _x_train, _x_vali, _y_train, _y_vali, _group_train, _group_test
        _BASE_PATH = self._BASE_PATH
        if _x_train is None:
            start = time()
            print('Loading data...\n')
            _x_train = pd.read_csv(f'{_BASE_PATH}/x_train.csv')
            _y_train = np.load(f'{_BASE_PATH}/y_train.npy')
            _group_train = np.load(f'{_BASE_PATH}/groups_train.npy')
            _x_vali = pd.read_csv(f'{_BASE_PATH}/x_vali.csv')
            _y_vali = np.load(f'{_BASE_PATH}/y_vali.npy')
            _group_test = np.load(f'{_BASE_PATH}/groups_vali.npy')
            print(f'data loaded in: {time() - start}\n')

        self.x_train = _x_train
        self.y_train = _y_train
        self.groups_train = _group_train

        self.x_vali = _x_vali
        self.y_vali = _y_vali
        self.groups_vali = _group_test


    def __init__(self, mode, cluster, dataset_name, params_dict):
        self.dataset_name = dataset_name
        super(lightGBM, self).__init__(
            name=f'lightGBM_{dataset_name}', mode=mode, cluster=cluster)
        self._BASE_PATH = f'dataset/preprocessed/lightGBM/{self.cluster}/{self.mode}/{self.dataset_name}'
        self._load_data()
        self.params_dict = params_dict
        self.eval_res = {}
        self.model = None

    def fit(self):
        # initialize the model
        self.model = lgb.LGBMRanker(**self.params_dict)
        self.model.fit(self.x_train, self.y_train, group=self.groups_train, verbose=False)

    def validate(self):
        def _mrr(y_true, y_pred, weight, group):
            def _order_unzip_pad(arr):
                # sort based on the score
                ordered_arr = sorted(arr, key=lambda x: x[1], reverse=True)
                # mantain only the label
                labels_ordered = np.array(ordered_arr)[:, 0]
                len_labels_ordered = len(labels_ordered)
                if len_labels_ordered < 25:
                    labels_padded = np.append(labels_ordered, np.zeros(25 - len_labels_ordered))
                else:
                    labels_padded = labels_ordered
                return labels_padded

            # define the fixed array weights
            WEIGHTS_ARR = 1 / np.arange(1, 26)

            # retrieve the indices where to split with a cumsim
            indices_split = np.cumsum(group)[:-1]

            # zip the score and the label
            couples_matrix_flattened = np.array(list(zip(y_true, y_pred)))

            # split with the group
            couples_matrix = np.split(couples_matrix_flattened, indices_split, axis=0)
            temp = np.array(list(map(_order_unzip_pad, couples_matrix)))
            mrr = np.sum(temp * WEIGHTS_ARR) / temp.shape[0]
            return 'MRR', mrr, True

        def _hera_callback(param):
            iteration_num = param[2]
            if iteration_num % param[1]['print_every'] == 0:
                message = f'PARAMS:\n'
                for k in param[1]:
                    message += f'{k}: {param[1][k]}\n'
                Hera.send_message(f'ITERATION_NUM: {iteration_num}\n {message}\n MRR: {param[5][0][2]}')

        # define a callback that will insert whitin the dictionary passed the history of the MRR metric during
        # the training phase
        eval_callback = lgb.record_evaluation(self.eval_res)

        # initialize the model
        self.model = lgb.LGBMRanker(**self.params_dict)

        self.model.fit(self.x_train, self.y_train, group=self.groups_train, eval_set=[(self.x_vali, self.y_vali)],
                  eval_group=[self.groups_vali], eval_metric=_mrr, eval_names='validation_set',
                  early_stopping_rounds=200, verbose=1, callbacks=[eval_callback, _hera_callback])
        # save the model parameters
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        check_folder(f'{self._BASE_PATH}/{time}')
        with open(f"{self._BASE_PATH}/{time}/Parameters.txt", "w+") as text_file:
            text_file.write(str(self.params_dict))
        self.model.booster_.save_model(f'{self._BASE_PATH}/{time}/{self.name}')
        # return negative mrr
        return self.eval_res['validation_set']['MRR'][self.model.booster_.best_iteration - 1]


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
    def get_optimize_params():
        space = [
            Real(0.05, 0.2, name='learning_rate'),
            Integer(6, 80, name='num_leaves'),
            Integer(100, 500, name='min_child_samples'),
            Real(1e-5, 1e4, name='min_child_weight'),
            Real(0.2, 0.8, name='subsample'),
            Real(0.4, 0.6, name='colsample_bytree'),
            Real(0, 0.5, name='reg_lambda'),
            Real(0, 0.5, name='reg_alpha'),
        ]

        def get_mrr(arg_list):

            learning_rate, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, reg_lambda, reg_alpha = arg_list
            Hera.send_message(f'Starting a train of bayesyan search with following params:\n '
                              f'learning_rate:{learning_rate}, num_leaves:{num_leaves}, min_child_samples: {min_child_samples},'
                              f'min_child_weight:{min_child_weight}, subsample:{subsample}, colsample_by_tree:{colsample_bytree}'
                              f'reg_lambda{reg_lambda}, reg_alpha:{reg_alpha}')
            params_dict = {
                'boosting_type': 'gbdt',
                'num_leaves': num_leaves,
                'max_depth': -1,
                'n_estimators': 5000,
                'learning_rate': learning_rate,
                'subsample_for_bin': 200000,
                'class_weights': None,
                'min_split_gain': 0.0,
                'min_child_weight': min_child_weight,
                'min_child_samples': min_child_samples,
                'subsample': subsample,
                'subsample_freq': 0,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'random_state': None,
                'n_jobs': -1,
                'silent': False,
                'importance_type': 'split',
                'metric': 'None',
                'print_every': 100,
            }
            model=lightGBM(mode='small', cluster='no_cluster', dataset_name='prova', params_dict=params_dict)
            mrr = model.validate()
            Hera.send_message(f'MRR: {mrr}\n'
                              f'params:\n'
                              f'learning_rate:{learning_rate}, num_leaves:{num_leaves}, '
                              f'reg_lambda{reg_lambda}, reg_alpha:{reg_alpha}')
            return -mrr
        return space, get_mrr



if __name__ == '__main__':
    params_dict = {
        'boosting_type':'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample_for_bin': 200000,
        'class_weights': None,
        'min_split_gain': 0.0,
        'min_child_weight': 0.001,
        'min_child_samples': 20,
        'subsample':1.0,
        'subsample_freq': 0,
        'colsample_bytree':1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': None,
        'n_jobs': -1,
        'silent': False,
        'importance_type': 'split',
        'metric': 'None',
        'print_every': 100,
    }
    model = lightGBM(mode='small', cluster='no_cluster', dataset_name='prova', params_dict=params_dict)
    model.validate()






