from abc import abstractmethod
from abc import ABC
import os
import data
import numpy as np
from utils.check_folder import check_folder
import preprocess

class ClusterizeBase(ABC):

    """
    A cluster is basically a set of session_ids to predict.
    Extending this base class is possible to specify a set of indices (ie of sessions)
    that a recommender should predict, other than create an ad-hoc dataset
    """

    def __init__(self, name):
        self.name = name

        self.train_indices = []
        self.test_indices = []
        self.target_indices = []

    @abstractmethod
    def _fit(self):
        """
        save on self the following LISTS:
        self.train_indices -> indices of base_split/full/train to be included in cluster full training set
        self.test_indices -> indices of base_split/full/test   to be included in cluster full test set
        self.target_indices -> indices of base_split/full/test that represents clickouts to be predicted
        
        third argument is optional: one should specify it only in case wants to impose the clickouts
                                    to be predicted. otherwise those will be set automatically equal
                                    to the missing clickouts indicated by the 2nd argument 
        """
        pass
    
    def save(self):
        """
        makes use of fit to create the dataset for a specific cluster. in particular it take cares
        to create a folder at the same level of base_split with the specified name and the
        folders structure inside 
        """
        self._fit()
        print('Indices retrievied correctly ...')

        full_df = data.full_df()

        # create cluster root folder
        path = f'dataset/preprocessed/{self.name}'
        check_folder(path)

        # create full and local folders
        full_path = os.path.join(path, 'full')
        local_path = os.path.join(path, 'local')
        small_path = os.path.join(path, 'small')
        check_folder(full_path)
        check_folder(local_path)
        check_folder(small_path)

        set_train_indices = set(self.train_indices)
        set_test_indices = set(self.test_indices)
        set_target_indices = set(self.target_indices)
        
        # create FULL files:    train_df.csv  -  test_df.csv  -  target_indices.npy
        test_full_df = full_df.loc[self.test_indices].to_csv(os.path.join(full_path, 'test.csv'))
        if len(self.target_indices) > 0:
            np.save(os.path.join(full_path, 'target_indices'), self.target_indices)
        else:
            trgt_indices = preprocess.get_target_indices(test_full_df)
            np.save(os.path.join(full_path, 'target_indices'), trgt_indices)
        del test_full_df

        train_full_df = full_df.loc[self.train_indices]
        train_full_df.to_csv(os.path.join(full_path, 'train.csv'))
        del train_full_df
        print('full cluster folder created...')

        # create LOCAL files:    train_df.csv  -  test_df.csv  -  target_indices.npy
        train_local_df = data.train_df('local')
        valid_train_indices = set(train_local_df.index).intersection(set_train_indices)
        train_local_df.loc[valid_train_indices].to_csv(os.path.join(local_path, 'train.csv'))
        del train_local_df
        del valid_train_indices

        test_local_df = data.test_df('local')
        valid_test_indices = set(test_local_df.index).intersection(set_test_indices)
        test_local_df = test_local_df.loc[valid_test_indices]
        test_local_df.to_csv(os.path.join(local_path, 'test.csv'))
        if len(self.target_indices) > 0:
            np.save(os.path.join(local_path, 'target_indices'), list(valid_test_indices & set_target_indices) )
        else:
            trgt_indices = preprocess.get_target_indices(test_local_df)
            np.save(os.path.join(local_path, 'target_indices'), trgt_indices)
        del test_local_df
        del valid_test_indices
        print('Local cluster folder created...')

        # create SMALL files:    train_df.csv  -  test_df.csv  -  target_indices.npy
        train_small_df = data.train_df('small')
        valid_train_indices = set(train_small_df.index).intersection(set_train_indices)
        train_small_df.loc[valid_train_indices].to_csv(os.path.join(small_path, 'train.csv'))
        del train_small_df
        test_small_df = data.test_df('small')
        valid_test_indices = set(test_small_df.index).intersection(set_test_indices)
        test_small_df = test_small_df.loc[valid_test_indices]
        test_small_df.to_csv(os.path.join(small_path, 'test.csv'))
        if len(self.target_indices) > 0:
            np.save(os.path.join(small_path, 'target_indices'), list(valid_test_indices & set_target_indices) )
        else:
            trgt_indices = preprocess.get_target_indices(test_small_df)
            np.save(os.path.join(small_path, 'target_indices'), trgt_indices)
        del test_small_df
        print('Small cluster folder created...')
