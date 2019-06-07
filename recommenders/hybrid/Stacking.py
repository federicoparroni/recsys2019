from recommenders.recommender_base import RecommenderBase
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
tqdm.pandas()
from pathlib import Path
import os
import data
import xgboost as xgb
from cython_files.mrr import mrr as mrr_cython
import utils.telegram_bot as HERA



class Stacking(RecommenderBase):

    def __init__(self, mode='local', learning_rate=0.3, min_child_weight=1, n_estimators=300, max_depth=3,
                 subsample=1, colsample_bytree=1, reg_lambda=1.0, reg_alpha=0):
        name = 'Stacking'
        cluster = 'no_cluster'
        super(Stacking, self).__init__(mode, cluster, name)

        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', '..', 'submissions')
        # self.gt_csv = self.data_directory.joinpath('ground_truth.csv')
        self.mode = mode
        self.xg = xgb.XGBRanker(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            n_jobs=-1, objective='rank:ndcg')

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'ask_to_load': False,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
        }

        global _best_so_far
        global _group_t
        global _kind
        _best_so_far = 0
        _group_t = []

    def create_groups_cv(self,df):
        df = df[['index', 'item_id']]
        group = df.groupby('index',
                           sort=False).apply(lambda x: len(x)).values
        return group

    def fit(self):
        global _group_t

        train = pd.read_csv(str(self.data_directory.joinpath('gbdt_train.csv')))
        val = pd.read_csv(str(self.data_directory.joinpath('gbdt_test.csv')))

        X_train = train.drop(
            ['item_id', 'label', 'index'], axis=1)
        X_val = val.drop(['item_id', 'label', 'index'], axis=1)

        print(f'Train columns: {list(X_train.columns)}')
        y_train = list(train['label'].values)
        y_val = list(val['label'].values)

        X_train = X_train.astype(np.float64)
        X_val = X_val.astype(np.float64)

        group = self.create_groups_cv(train)
        group_val = self.create_groups_cv(val)

        _group_t = group_val

        print('Training XGBOOST..')
        self.xg.fit(X_train, y_train, group, eval_set=[
                    (X_val, y_val)], eval_group=[group_val], eval_metric=_mrr, verbose=True, callbacks=[callbak], early_stopping_rounds=200)

        print('Training done!')

    def evaluate(self):
        self.fit()
        results = self.xg.evals_result
        MRRs = -np.array(results['eval_0']['MRR'])
        max_mrr = np.amax(MRRs)
        max_idx = np.argmax(MRRs)
        self.fixed_params_dict['n_estimators'] = max_idx
        return max_mrr

    def recommend_batch(self):
        pass

    def get_scores_batch(self):
        pass

_best_so_far = 0
_group_t = []
_kind = ''

def callbak(obj):
    global _best_so_far
    if -obj[6][1][1] > _best_so_far:
        _best_so_far = -obj[6][1][1]
        #if _best_so_far > 0.67:
        #    HERA.send_message('xgboost {} iteration {} mrr is {}'. format(
        #        _kind, obj.iteration, _best_so_far), 'ale')
        print('xgboost iteration {} mrr is {}'. format(obj.iteration, _best_so_far))


def _mrr(y_true, y_pred):
    y_pred = y_pred.get_label()
    l = memoryview(np.array(y_pred, dtype=np.int32))
    p = memoryview(np.array(y_true, dtype=np.float32))
    g = memoryview(np.array(_group_t, dtype=np.int32))
    mrr = mrr_cython(l, p, g, len(_group_t))
    return 'MRR', -mrr


if __name__ == '__main__':
    model = Stacking(mode='local')
    model.fit()
    print(model.evaluate())