from utils.menu import yesno_choice
from utils.check_folder import check_folder
import os
import math
import xgboost as xgb
import data
from recommenders.recommender_base import RecommenderBase
import numpy as np
import pandas as pd
from pandas import Series
from tqdm import tqdm
import scipy.sparse as sps
tqdm.pandas()
import utils.telegram_bot as HERA
from cython_files.mrr import mrr as mrr_cython


class XGBoostWrapper(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster', kind='kind1', ask_to_load=True, class_weights=False, learning_rate=0.3, min_child_weight=1, n_estimators=100, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        name = 'xgboost_ranker_mode={}_cluster={}_kind={}_class_weights={}_learning_rate={}_min_child_weight={}_n_estimators={}_max_depth={}_subsample={}_colsample_bytree={}_reg_lambda={}_reg_alpha={}'.format(
            mode, cluster, kind, class_weights, learning_rate, min_child_weight, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda, reg_alpha
        )
        super(XGBoostWrapper, self).__init__(
            name=name, mode=mode, cluster=cluster)
        self.class_weights = class_weights
        self.kind = kind
        self.ask_to_load = ask_to_load

        self.xg = xgb.XGBRanker(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha, n_jobs=-1, objective='rank:pairwise')

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'kind': kind,
            'ask_to_load': False,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'learning_rate': (0.01, 0.3),
                                     'max_depth': (3, 7),
                                     'n_estimators': (700, 1200),
                                     'reg_lambda': (0, 0.5),
                                     'reg_alpha': (0, 0.5)
                                     }

    def fit(self):
        check_folder('models')
        if self.ask_to_load:
            if os.path.isfile('models/{}.model'.format(self.name)):
                if yesno_choice('the exact same model was yet created. want to load?') == 'y':
                    self.xg.load_model('models/{}.model'.format(self.name))
                    return

        if self.class_weights:
            X_train, y_train, group, _, weights, _ = data.dataset_xgboost_train(
                mode=self.mode, cluster=self.cluster, class_weights=self.class_weights, kind=self.kind)
        else:
            X_train, y_train, group, _, _ = data.dataset_xgboost_train(
                mode=self.mode, cluster=self.cluster, class_weights=self.class_weights, kind=self.kind)
        print('data for train ready')

        if self.class_weights:
            self.xg.fit(X_train, y_train, group, sample_weight=weights)
        else:
            self.xg.fit(X_train, y_train, group)
        print('fit done')
        self.xg.save_model('models/{}.model'.format(self.name))
        print('model saved')

    def recommend_batch(self):
        X_test, _, _, _ = data.dataset_xgboost_test(
            mode=self.mode, cluster=self.cluster, kind=self.kind)
        target_indices = data.target_indices(self.mode, self.cluster)
        full_impressions = data.full_df()
        print('data for test ready')
        scores = list(self.xg.predict(X_test))
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
            count = count+len(impressions)
        return final_predictions

    def get_scores_batch(self):
        X_test, _, _, _ = data.dataset_xgboost_test(
            mode=self.mode, cluster=self.cluster, kind=self.kind)
        target_indices = data.target_indices(self.mode, self.cluster)
        full_impressions = data.full_df()
        print('data for scores test ready')
        scores = list(self.xg.predict(X_test))
        final_predictions_with_scores_test = []
        count = 0
        for index in tqdm(target_indices):
            impressions = list(
                map(int, full_impressions.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            sorted_scores, sorted_impr = zip(*couples)
            final_predictions_with_scores_test.append(
                (index, list(sorted_impr), list(sorted_scores)))
            count = count + len(impressions)

        X_train, _, _, train_indices, _ = data.dataset_xgboost_train(
            mode=self.mode, cluster=self.cluster, kind=self.kind)
        full_impressions = data.full_df()
        print('data for scores train ready')
        scores = list(self.xg.predict(X_train))
        final_predictions_with_scores_train = []
        count = 0
        for index in tqdm(train_indices):
            impressions = list(
                map(int, full_impressions.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            sorted_scores, sorted_impr = zip(*couples)
            final_predictions_with_scores_train.append(
                (index, list(sorted_impr), list(sorted_scores)))
            count = count + len(impressions)

        return final_predictions_with_scores_train, final_predictions_with_scores_test

    def compute_MRR(self, predictions):
        """
        :param predictions:
        :return: MRR computed on just the sessions where the clickout is not on the first impression
        """
        assert (self.mode == 'local' or self.mode == 'small')
        train_df = pd.read_csv('dataset/preprocessed/no_cluster/full/train.csv'.format(
            self.cluster), usecols=['reference', 'impressions'])

        target_indices, recs = zip(*predictions)
        target_indices = list(target_indices)
        correct_clickouts = train_df.loc[target_indices].reference.values
        impression = train_df.loc[target_indices].impressions.values
        len_rec = len(recs)
        count = 0

        RR = 0
        print("Calculating MRR (hoping for a 0.99)")
        for i in tqdm(range(len_rec)):
            if impression[i].split('|').index(correct_clickouts[i]) != 0 or not self.class_weights:
                correct_clickout = int(correct_clickouts[i])
                if correct_clickout in predictions[i][1]:
                    rank_pos = recs[i].index(correct_clickout) + 1
                    if rank_pos <= 25:
                        RR += 1 / rank_pos
                count += 1
            else:
                print('skipping because:')
                print(impression[i])
                print(correct_clickouts[i])

        MRR = RR / count
        print(f'MRR: {MRR}')

        return MRR

    def fit_cv(self, x, y, groups, train_indices, test_indices, **fit_params):
        X_train = x[train_indices, :]
        y_train = y.loc[train_indices]
        _, group = np.unique(groups[train_indices], return_counts=True)
        self.xg.fit(X_train, y_train, group)

    def get_scores_cv(self, x, groups, test_indices):
        if x.shape[0] == len(test_indices):
            _, _, _, user_session_item = data.dataset_xgboost_test(mode=self.mode, cluster='no_cluster', kind=self.kind)
        else:
            _, _, _, _, user_session_item = data.dataset_xgboost_train(mode=self.mode, cluster='no_cluster', kind=self.kind)
        
        X_test = x[test_indices, :]
        preds = list(self.xg.predict(X_test))
        user_session_item = user_session_item.loc[test_indices]
        user_session_item['score_xgboost'] = preds
        return user_session_item
 
class XGBoostWrapperSmartValidation(XGBoostWrapper):

    def __init__(self, mode, cluster='no_cluster', kind='kind1', ask_to_load=True, class_weights=False, learning_rate=0.3, min_child_weight=1, n_estimators=100, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        super(XGBoostWrapperSmartValidation, self).__init__(mode, cluster=cluster, kind=kind, ask_to_load=False, class_weights=False,
                                                            learning_rate=learning_rate, min_child_weight=min_child_weight,
                                                            n_estimators=n_estimators, max_depth=max_depth, subsample=subsample,
                                                            colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha)
        self.name = 'lr={} min_child_weight={} n_estimators={}, max_depth={}, subsample={}, colsample_bytree={}, reg_lambda={}, reg_alpha={}'.format(
            learning_rate, min_child_weight, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda, reg_alpha)

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'kind': kind,
            'ask_to_load': False,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
            'n_estimators': 100000,
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'learning_rate': (0.1, 0.2),
                                     'max_depth': (6, 9),
                                     'reg_lambda': (2, 4),
                                     'reg_alpha': (7, 10)
                                     }
        global _best_so_far
        global _group_t
        global _kind
        _best_so_far = 0
        _group_t = []
        _kind = kind

    def fit(self):
        global _group_t
        X_train, y_train, group, _, _ = data.dataset_xgboost_train(
            mode=self.mode, cluster=self.cluster, class_weights=self.class_weights, kind=self.kind)
        print('data for train ready')

        X_test, y_test, groups_test, _ = data.dataset_xgboost_test(
            mode=self.mode, cluster=self.cluster, kind=self.kind)
        _group_t = groups_test
        print('data for evaluation ready')

        self.xg.fit(X_train, y_train, group, eval_set=[
                    (X_test, y_test)], eval_group=[groups_test], eval_metric=_mrr, verbose=False, callbacks=[callbak],
                    early_stopping_rounds=200)

    def evaluate(self):
        self.fit()
        results = self.xg.evals_result
        MRRs = -np.array(results['eval_0']['MRR'])
        max_mrr = np.amax(MRRs)
        max_idx = np.argmax(MRRs)
        self.fixed_params_dict['n_estimators'] = max_idx
        return max_mrr

_best_so_far = 0
_group_t = []
_kind = ''

def callbak(obj):
    global _best_so_far
    if -obj[6][1][1] > _best_so_far:
        _best_so_far = -obj[6][1][1]
        if _best_so_far > 0.67:
            HERA.send_message('xgboost {} iteration {} mrr is {}'. format(
                _kind, obj.iteration, _best_so_far), 'edo')
        print('xgboost iteration {} mrr is {}'. format(obj.iteration, _best_so_far))


def _mrr(y_true, y_pred):
    y_pred = y_pred.get_label()
    l = memoryview(np.array(y_pred, dtype=np.int32))
    p = memoryview(np.array(y_true, dtype=np.float32))
    g = memoryview(np.array(_group_t, dtype=np.int32))
    mrr = mrr_cython(l, p, g, len(_group_t))
    return 'MRR', -mrr


if __name__ == '__main__':
    from utils.menu import mode_selection
    from utils.menu import cluster_selection
    from utils.menu import single_choice
    from utils.menu import options

    modality = single_choice('smart evaluate or normal recommender?', [
                             'smart evaluate', 'normal recommender'])

    if modality == 'normal recommender':
        kind = single_choice('pick the kind', ['kind1', 'kind2', 'kind3'])
        mode = mode_selection()
        cluster = cluster_selection()
        sel = options(['evaluate', 'export the sub', 'export the scores'], ['evaluate', 'export the sub',
                                                                            'export the scores'], 'what do you want to do after model fitting and the recommendations?')
        model = XGBoostWrapper(mode=mode, cluster=cluster, kind=kind)
        if 'evaluate' in sel:
            model.evaluate(True)
        if 'export the sub' in sel and 'export the scores' in sel:
            model.run(export_sub=True, export_scores=True)
        elif 'export the sub' in sel and 'export the scores' not in sel:
            model.run(export_sub=True, export_scores=False)
        elif 'export the sub' not in sel and 'export the scores' in sel:
            model.run(export_sub=False, export_scores=True)

    else:
        mode = mode_selection()
        cluster = cluster_selection()
        model = XGBoostWrapperSmartValidation(mode=mode, cluster=cluster)
        model.fit()
        print(model.evaluate())
