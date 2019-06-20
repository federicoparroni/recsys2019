import data
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import utils.menu as menu
from utils.check_folder import check_folder
import os.path


class ScoresCV(object):

    def __init__(self, model_class, init_params):
        self.model_class = model_class
        self.init_params = init_params
        self.scores = []


    def _fit_model(self, i):
        mode = 'local'
        cluster = 'fold_' + str(i)
        model = self.model_class(mode=mode, cluster=cluster, **self.init_params)
        model.fit()
        scores = model.get_scores_batch()
        return scores

    def _fit_predict(self, save_folder='scores/'):
        self.scores = Parallel(backend='multiprocessing', n_jobs=-1, max_nbytes=None)(delayed(self._fit_model)(
            i
        ) for i in range(5))


        print(len(self.scores))

        model = self.model_class(mode='full', cluster='no_cluster', **self.init_params)
        model.fit()
        scores_test = model.get_scores_batch()
        self.scores.append(scores_test)

        print(len(self.scores))

        # if save_folder is not None:
        #     check_folder(save_folder)
        #     filepath = os.path.join(save_folder, model.name + '.csv.gz')
        #     print('Saving scores to', filepath, end=' ', flush=True)
        #     self.scores.to_csv(filepath, index=False, compression='gzip')
        #     print('Done!', flush=True)

        return self.scores


if __name__=='__main__':
    pass

