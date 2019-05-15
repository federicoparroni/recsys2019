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
import os
from utils.check_folder import check_folder
from utils.menu import yesno_choice


class XGBoostWrapper(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster', class_weights=False, learning_rate=0.3, min_child_weight=1, n_estimators=100, max_depth=3, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0):
        name = 'xgboost_ranker_mode={}_cluster={}_class_weights={}_learning_rate={}_min_child_weight={}_n_estimators={}_max_depth={}_subsample={}_colsample_bytree={}_reg_lambda={}_reg_alpha={}'.format(
            mode, cluster, class_weights, learning_rate, min_child_weight, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda, reg_alpha
        )
        super(XGBoostWrapper, self).__init__(
            name=name, mode=mode, cluster=cluster)
        self.class_weights = class_weights

        self.xg = xgb.XGBRanker(
            learning_rate=learning_rate, min_child_weight=min_child_weight, max_depth=math.ceil(
                max_depth),
            n_estimators=math.ceil(
                n_estimators),
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha, n_jobs=-1, objective='rank:pairwise')

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1,
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'learning_rate': (0.01, 0.3),
                                     'max_depth': (2, 6),
                                     'n_estimators': (300, 1000),
                                     'reg_lambda': (0, 1),
                                     'reg_alpha': (0, 1)
                                     }

    def fit(self):
        check_folder('models')
        if os.path.isfile('models/{}.model'.format(self.name)):
            if yesno_choice('the exact same model was yet created. want to load?') == 'y':
                self.xg.load_model('models/{}.model'.format(self.name))
                return

        if self.class_weights:
            X_train, y_train, group, weights = data.dataset_xgboost_train(
                mode=self.mode, cluster=self.cluster, class_weights=self.class_weights)
        else:
            X_train, y_train, group = data.dataset_xgboost_train(
                mode=self.mode, cluster=self.cluster, class_weights=self.class_weights)
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
            mode=self.mode, cluster=self.cluster)
        target_indices = data.target_indices(self.mode, self.cluster)
        full_impressions = pd.read_csv(
            'dataset/preprocessed/full.csv', usecols=["impressions"])
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
        full_impressions = pd.read_csv(
            'dataset/preprocessed/full.csv', usecols=["impressions"])
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
        train_df = pd.read_csv('dataset/preprocessed/{}/full/train.csv'.format(self.cluster), usecols=['reference', 'impressions'])

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
                print('class weights: {}'.format(self.class_weights))

        MRR = RR / count
        print(f'MRR: {MRR}')

        return MRR

if __name__ == '__main__':
    from utils.menu import mode_selection
    import out
    mode = mode_selection()
    model = XGBoostWrapper(mode=mode, cluster='no_cluster')
    model.fit()
    recs = model.recommend_batch()
    MRR = model.compute_MRR(recs)
    #out.create_sub(recs, submission_name=model.name)
    # model.evaluate(send_MRR_on_telegram=True)
    # model.run(False)
