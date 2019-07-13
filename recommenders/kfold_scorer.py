import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import copy
from utils.check_folder import check_folder
from sklearn.model_selection import KFold
from joblib import Parallel, delayed


class KFoldScorer(object):
    """
    Get the scores for the dataset by fitting each model in K-fold (except one) and
    computing the scores for the left-out fold.
    The underlying model should implement the following methods:
    - fit_cv(x, y, groups, train_indices, test_indices, **fit_params)
    - get_scores_cv(x, groups, test_indices)   : must return a dataframe with columns [ user_id | session_id | item_id | score ] 
    """

    def __init__(self, model_class, init_params:dict, k:int):
        self.model_class = model_class
        self.init_params = init_params
        self.k = k
        self.scores = []
    
    # train a single model on a fold
    def _fit_model(self, x, y, groups, train_indices, test_indices, fit_params, pool_id=0):
        model = self.model_class(**self.init_params)
        print(f'start {pool_id} on {model.name}')
        assert hasattr(model, 'fit_cv') and hasattr(model, 'get_scores_cv'), \
            'Model must implement methods: fit_cv, get_scores_cv'
        model.fit_cv(x, y, groups, train_indices, test_indices, **fit_params)
        print(f'fit end {pool_id}')
        # compute scores
        return model.get_scores_cv(x, groups, test_indices)

    def fit_predict(self, dataset, fit_params={}, multithreading=True, n_jobs=-1, save_folder='scores/') -> pd.DataFrame:
        """ Fit and compute the scores for each fold.
        dataset (object):   inheriting from utils.dataset.DatasetBase
        fit_params (dict):  params to fit the model
        n_jobs (int):       maximum number of concurrently running jobs, -1 to use all CPUs
        save_folder (str):  folder where to save the scores
        """
        assert hasattr(dataset, 'load_Xtrain') and hasattr(dataset, 'load_Ytrain') and hasattr(dataset, 'load_Xtest'), \
                    'Dataset object must implement methods: load_Xtrain, load_Ytrain, load_Xtest'
        
        X_train, Y_train, X_test, group_train = dataset.load_Xtrain(), dataset.load_Ytrain(), \
                                                dataset.load_Xtest(), dataset.load_group_train()
        
        # kfold
        kf = KFold(n_splits=self.k)
        
        # fit in each fold
        if multithreading:
            self.scores = Parallel(backend='multiprocessing', n_jobs=n_jobs)(delayed(self._fit_model)
                                (
                                    X_train, Y_train, group_train,
                                    train_indices, test_indices,
                                    fit_params, idx
                                ) for idx,(train_indices,test_indices) in enumerate(kf.split(X_train, group_train)) )
        else:
            self.scores = [self._fit_model
                                (
                                    X_train, Y_train, group_train,
                                    train_indices, test_indices,
                                    fit_params, idx
                                ) for idx,(train_indices,test_indices) in enumerate(kf.split(X_train, group_train)) ]
        
        # fit in all the train and get scores for test
        print('fit whole train')
        model = self.model_class(**self.init_params)
        model.fit_cv(X_train, Y_train, group_train, list(range(X_train.shape[0])), [], **fit_params)
        print('end fit whole train')
        scores_test = model.get_scores_cv(X_test, None, list(range(X_test.shape[0])))
        self.scores.append( scores_test )
        
        self.scores = pd.concat(self.scores)

        # save scores
        if save_folder is not None:
            check_folder(save_folder)
            filepath = os.path.join(save_folder, model.name + '.csv.gz')
            print('Saving scores to', filepath, end=' ', flush=True)
            self.scores.to_csv(filepath, index=False, compression='gzip')
            print('Done!', flush=True)
        
        return self.scores
        

if __name__ == "__main__":
    import utils.menu as menu
    from utils.dataset import DatasetScoresClassification, DatasetScoresBinaryClassification
    from recommenders.recurrent.RNNClassificationRecommender import RNNClassificationRecommender
    from recommenders.recurrent.RNNBinaryClassificator import RNNBinaryClassificator
    
    #from keras.optimizers import Adam

    def scores_rnn():
        dataset = DatasetScoresClassification(f'dataset/preprocessed/cluster_recurrent/full/dataset_classification_p12')

        init_params = {
            'dataset': dataset,
            'input_shape': (12,168),
            'cell_type': 'gru',
            'num_recurrent_layers': 2,
            'num_recurrent_units': 64,
            'num_dense_layers': 2,
            'optimizer': 'adam',
            #'class_weights': dataset.get_class_weights(),
            #'sample_weights': dataset.get_sample_weights()
        }
        fit_params = {'epochs': int(input('Insert number of epochs: '))}

        kfscorer = KFoldScorer(model_class=RNNClassificationRecommender, init_params=init_params, k=5)

        kfscorer.fit_predict(dataset, multithreading=True, fit_params=fit_params)

    
    def scores_bin():
        dataset = DatasetScoresBinaryClassification(f'dataset/preprocessed/cluster_recurrent/small/dataset_binary_classification_p6')

        init_params = {
            'dataset': dataset,
            'input_shape': (6,168),
            'cell_type': 'gru',
            'num_recurrent_layers': 2,
            'num_recurrent_units': 64,
            'num_dense_layers': 2,
            'optimizer': 'adam',
            #'class_weights': dataset.get_class_weights(),
            'sample_weights': dataset.get_sample_weights()
        }
        fit_params = {'epochs': int(input('Insert number of epochs: '))}

        kfscorer = KFoldScorer(model_class=RNNBinaryClassificator, init_params=init_params, k=5)

        kfscorer.fit_predict(dataset, multithreading=True, fit_params=fit_params)

    menu.single_choice('Which model?', ['RNN', 'RNN binary'], [scores_rnn, scores_bin])
