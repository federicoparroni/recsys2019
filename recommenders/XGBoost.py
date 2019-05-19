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


class XGBoostWrapper(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster', kind='kind1', class_weights=False, learning_rate=0.3, min_child_weight=1, n_estimators=100, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        name = 'xgboost_ranker_mode={}_cluster={}_kind={}_class_weights={}_learning_rate={}_min_child_weight={}_n_estimators={}_max_depth={}_subsample={}_colsample_bytree={}_reg_lambda={}_reg_alpha={}'.format(
            mode, cluster, kind, class_weights, learning_rate, min_child_weight, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda, reg_alpha
        )
        super(XGBoostWrapper, self).__init__(
            name=name, mode=mode, cluster=cluster)
        self.class_weights = class_weights
        self.kind = kind

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
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'learning_rate': (0.01, 0.3),
                                     'max_depth': (3, 7),
                                     'n_estimators': (700, 1500),
                                     'reg_lambda': (0, 0.5),
                                     'reg_alpha': (0, 0.5)
                                     }

    def fit(self):
        check_folder('models')
        if os.path.isfile('models/{}.model'.format(self.name)):
            if yesno_choice('the exact same model was yet created. want to load?') == 'y':
                self.xg.load_model('models/{}.model'.format(self.name))
                return

        if self.class_weights:
            X_train, y_train, group, weights = data.dataset_xgboost_train(
                mode=self.mode, cluster=self.cluster, class_weights=self.class_weights, kind=kind)
        else:
            X_train, y_train, group = data.dataset_xgboost_train(
                mode=self.mode, cluster=self.cluster, class_weights=self.class_weights, kind=kind)
        print('data for train ready')

        if self.class_weights:
            self.xg.fit(X_train, y_train, group, sample_weight=weights)
        else:
            self.xg.fit(X_train, y_train, group)
        print('fit done')
        self.xg.save_model('models/{}.model'.format(self.name))
        print('model saved')

    def recommend_batch(self):
        X_test = data.dataset_xgboost_test(
            mode=self.mode, cluster=self.cluster, kind=kind)
        target_indices = data.target_indices(self.mode, self.cluster)
        # full_impressions = pd.read_csv(
        #     'dataset/preprocessed/full.csv', usecols=["impressions"])
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
        X_test = data.dataset_xgboost_test(
            mode=self.mode, cluster=self.cluster)
        target_indices = data.target_indices(self.mode, self.cluster)
        # full_impressions = pd.read_csv(
        #     'dataset/preprocessed/full.csv', usecols=["impressions"])
        full_impressions = data.full_df()
        print('data for test ready')
        scores = list(self.xg.predict(X_test))
        final_predictions_with_scores = []
        count = 0
        for index in tqdm(target_indices):
            impressions = list(
                map(int, full_impressions.loc[index]['impressions'].split('|')))
            predictions = scores[count:count + len(impressions)]
            couples = list(zip(predictions, impressions))
            couples.sort(key=lambda x: x[0], reverse=True)
            sorted_scores, sorted_impr = zip(*couples)
            final_predictions_with_scores.append(
                (index, list(sorted_impr), list(sorted_scores)))
            count = count + len(impressions)
        return final_predictions_with_scores

    def compute_MRR(self, predictions):
        """
        :param predictions:
        :return: MRR computed on just the sessions where the clickout is not on the first impression
        """
        assert (self.mode == 'local' or self.mode == 'small')
        train_df = pd.read_csv('dataset/preprocessed/{}/full/train.csv'.format(
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


if __name__ == '__main__':
    from utils.menu import mode_selection
    from utils.menu import single_choice
    from utils.menu import options
    kind = single_choice(['1', '2'], ['kind1', 'kind2'])
    mode = mode_selection()
    sel = options(['evaluate', 'export the sub', 'export the scores'], ['evaluate', 'export the sub',
                                                                  'export the scores'], 'what do you want to do after model fitting and the recommendations?')
    model = XGBoostWrapper(mode=mode, cluster='no_cluster', kind=kind)
    if 'evaluate' in sel:
        model.evaluate(True)
    if 'export the sub' in sel and 'export the scores' in sel:
        model.run(export_sub=True, export_scores=True)
    elif 'export the sub' in sel and 'export the scores' not in sel:
        model.run(export_sub=True, export_scores=False)
    elif 'export the sub' not in sel and 'export the scores' in sel:
        model.run(export_sub=False, export_scores=True)
