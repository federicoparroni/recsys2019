from abc import abstractmethod
from abc import ABC
import os
import data
import numpy as np
from utils.check_folder import check_folder
import preprocess
import pandas as pd

class ClusterizeBase(ABC):

    """
    A cluster is basically a set of session_ids to predict.
    Extending this base class is possible to specify a set of indices (ie of sessions)
    that a recommender should predict, other than create an ad-hoc dataset
    """

    def __init__(self, name='clusterize_base'):
        self.name = name

        self.train_indices = []
        self.test_indices = []
        self.target_indices = []

    @abstractmethod
    def _fit(self, mode):
        """
        save on self the following LISTS:
        self.train_indices -> indices of base_split/mode/train to be included in cluster full training set
        self.test_indices -> indices of base_split/mode/test   to be included in cluster full test set
        self.target_indices -> indices of base_split/mode/test that represents clickouts to be predicted
        
        third argument is optional: one should specify it only in case wants to impose the clickouts
                                    to be predicted. otherwise those will be set automatically equal
                                    to the missing clickouts indicated by the 2nd argument
        add_unused_clickouts_to_test: if True, not predicted clickouts are re-added to test in order to
                                    exploit another sort of information
        """
        pass
    
    def save(self, mode='full', add_unused_clickouts_to_test=True):
        """
        makes use of fit to create the dataset for a specific cluster. in particular it take cares
        to create a folder at the same level of base_split with the specified name and the
        folders structure inside 
        """
        print('Creating {} cluster...'.format(mode), end=' ', flush=True)
        self._fit(mode)

        # create cluster root folder
        path = f'dataset/preprocessed/{self.name}'
        check_folder(path)

        # create full and local folders
        full_path = os.path.join(path, mode)
        check_folder(full_path)

        train = data.train_df(mode).loc[self.train_indices]
        train.to_csv(os.path.join(full_path, 'train.csv'))
        del train

        # in case I specify some target_indices, I do not want to leave missing clickout not-to-predict
        if add_unused_clickouts_to_test & len(self.target_indices) > 0:
            indices_from_full = list(set(self.test_indices) - set(self.target_indices))
            indices_from_test = self.target_indices
            test = pd.concat([data.test_df(mode).loc[indices_from_test], data.full_df().loc[indices_from_full]])
        else:
            test = data.test_df(mode).loc[self.test_indices]

        test.to_csv(os.path.join(full_path, 'test.csv'))

        if len(self.target_indices) > 0:
            np.save(os.path.join(full_path, 'target_indices'), self.target_indices)
        else:
            trgt_indices = preprocess.get_target_indices(test)
            np.save(os.path.join(full_path, 'target_indices'), trgt_indices)
        del test

        print('Done!')

    def create(self):
        """
        call it on an object to create all the 3 clusters induced by that object
        """
        self.save('full')
        self.save('local')
        self.save('small')