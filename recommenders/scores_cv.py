import data
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import utils.menu as menu
from utils.check_folder import check_folder
import os.path
from preprocess_utils.extract_scores import assign_score


class ScoresCV(object):

    def __init__(self, mode, model_class, model_name,  init_params):
        """

        :param mode:
        :param model_class:
        :param model_name: name to insert in the score column: model_name = 'xgb' --> score_xgb
        :param init_params:
        """
        self.model_class = model_class
        self.init_params = init_params
        self.model_name = model_name
        self.scores = []
        self.mode = mode

    def _fit_model(self, i):
        cluster = 'fold_' + str(i)
        model = self.model_class(mode=self.mode, cluster=cluster, **self.init_params)
        model.fit()
        scores = model.get_scores_batch()
        return scores

    def fit_predict(self, multithreading = True, save_folder='scores/'):

        if multithreading:
            self.scores = Parallel(backend='multiprocessing', n_jobs=-1, max_nbytes=None)(delayed(self._fit_model)(
               i
            ) for i in range(5))

            print(len(self.scores))
        else:
            self.scores = [self._fit_model(i) for i in range(5)]
            print(len(self.scores))

        model = self.model_class(mode=self.mode, cluster='no_cluster', **self.init_params)
        model.fit()
        scores_test = model.get_scores_batch()
        self.scores.append(scores_test)

        self.scores = [item for sublist in self.scores for item in sublist]
        scores = pd.DataFrame(self.scores, columns=['index', 'item_recommendations','scores'])
        scores = scores.sort_values(by='index')
        print(scores)
        idx_scores = set(scores['index'].values)

        train_full = data.train_df(mode='full', cluster='no_cluster')
        test_full = data.test_df(mode='full', cluster='no_cluster')
        full = pd.concat([train_full, test_full])
        full = full[['user_id', 'session_id', 'action_type']]

        last_clk_full = full.loc[idx_scores]

        # checking that all rows are clickouts
        num_not_clk_row = last_clk_full[last_clk_full['action_type'] != 'clickout item'].shape[0]
        print(f'Number of not clickout rows is : {num_not_clk_row}')
        if num_not_clk_row != 0:
            print("Error, some indices are not clickouts")

        last_clk_full = last_clk_full.drop(['action_type'], axis=1)

        last_clk_full['index'] = last_clk_full.index
        merged = last_clk_full.merge(scores, on=['index'])
        model_name = model.name
        df = assign_score(merged, self.model_name)
        df = df.drop(['index'], axis=1)

        if save_folder is not None:
            check_folder(save_folder)
            filepath = os.path.join(save_folder, model_name + '.csv.gz')
            print('Saving scores to', filepath, end=' ', flush=True)
            df.to_csv(filepath, index=False, compression='gzip')
            print('Done!', flush=True)

        return df

if __name__=='__main__':
    from recommenders.XGBoost import XGBoostWrapper
    import utils.menu as menu

    mode = menu.mode_selection()
    kind = input('pick the kind: ')

    init_params = {
        'kind' : kind,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
        'learning_rate': 0.01,
        'max_depth': 3,
         'n_estimators': 100,
         'reg_lambda': 1.0,
         'reg_alpha': 0.0
         }

    scoresCV = ScoresCV(mode=mode, model_class=XGBoostWrapper, init_params=init_params)
    scoresCV.fit_predict(multithreading=False)




