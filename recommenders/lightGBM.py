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

def recommend_batch(pred):
    target_indices = data.target_indices('small')
    # full_impressions = pd.read_csv(
    #     'dataset/preprocessed/full.csv', usecols=["impressions"])
    full_impressions = data.full_df()
    print('data for test ready')
    scores = pred
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

class lightGBM(RecommenderBase):
    def _load_data(self):
        print('Loading data...\n')
        start = time()
        _BASE_PATH = f'dataset/preprocessed/lightGBM/{self.cluster}/{self.mode}/{self.dataset_name}'
        self.x_train = sps.load_npz(f'{_BASE_PATH}/x_train.csv')
        self.y_train = np.load(f'{_BASE_PATH}/y_train.npy')
        self.groups_train = np.load(f'{_BASE_PATH}/groups_train.npy')

        self.x_vali = sps.load_npz(f'{_BASE_PATH}/x_vali.csv')
        self.y_vali = np.load(f'{_BASE_PATH}/y_vali.npy')
        self.groups_vali = np.load(f'{_BASE_PATH}/groups_vali.npy')
        print(f'data loaded in: {time()-start}\n')

    def __init__(self, mode, cluster, dataset_name, params_dict):
        self.dataset_name = dataset_name
        super(lightGBM, self).__init__(
            name=f'lightGBM_{dataset_name}', mode=mode, cluster=cluster)
        self._load_data()

    def fit(self):
        pass

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
    'importance_type': 'split'
}


def _load_data(mode, cluster, dataset_name):
    _BASE_PATH = f'dataset/preprocessed/lightGBM/{cluster}/{mode}/{dataset_name}'
    x_train = pd.read_csv(f'{_BASE_PATH}/x_train.csv')
    y_train = np.load(f'{_BASE_PATH}/y_train.npy')
    groups_train = np.load(f'{_BASE_PATH}/groups_train.npy')

    x_vali = pd.read_csv(f'{_BASE_PATH}/x_vali.csv')
    y_vali = np.load(f'{_BASE_PATH}/y_vali.npy')
    groups_vali = np.load(f'{_BASE_PATH}/groups_vali.npy')

    return x_train, y_train, groups_train, x_vali, y_vali, groups_vali

def _mrr(y_true, y_pred, weight, group):
    def _order_unzip_pad(arr):
        # sort based on the score
        ordered_arr = sorted(arr, key=lambda x: x[1], reverse=True)
        # mantain only the label
        labels_ordered = np.array(ordered_arr)[:, 0]
        len_labels_ordered = len(labels_ordered)
        if len_labels_ordered < 25:
            labels_padded = np.append(labels_ordered, np.zeros(25-len_labels_ordered))
        else:
            labels_padded = labels_ordered
        return labels_padded

    # define the fixed array weights
    WEIGHTS_ARR = 1/np.arange(1, 26)

    # retrieve the indices where to split with a cumsim
    indices_split = np.cumsum(group)[:-1]

    # zip the score and the label
    couples_matrix_flattened = np.array(list(zip(y_true, y_pred)))

    # split with the group
    couples_matrix = np.split(couples_matrix_flattened, indices_split, axis=0)
    temp = np.array(list(map(_order_unzip_pad, couples_matrix)))
    mrr = np.sum(temp * WEIGHTS_ARR) / temp.shape[0]
    return 'MRR', mrr, True

x_train, y_train, groups_train, x_vali, y_vali, groups_vali = _load_data('small', 'no_cluster', 'prova')

e_res = {}
eval_callback = lgb.record_evaluation(e_res)

def custom_callback(param):
    iteration_num = param[2]
    if iteration_num % param[1]['print_every'] == 0:
        message = f'PARAMS:\n'
        for k in param[1]:
            message += f'{k}: {param[1][k]}\n'


        Hera.send_message(f'ITERATION_NUM: {iteration_num}\n {message}\n MRR: {param[5][0][2]}')


model = lgb.LGBMRanker(silent=False, boosting_type='gbdt', objective='lambdarank', n_estimators=10000000,
                       max_depth=7, num_leaves=21,**{'metric': 'None', 'print_every':20})
model.fit(x_train, y_train, group=groups_train, eval_set=[(x_vali, y_vali)], eval_group=[groups_vali],
          eval_metric=_mrr, eval_at=1, early_stopping_rounds=50, verbose=False, callbacks=[eval_callback, custom_callback])



