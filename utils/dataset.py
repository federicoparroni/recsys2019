import os
import math
import pandas as pd
import numpy as np
from abc import abstractmethod

import utils.datasetconfig as datasetconfig
from generator import DataGenerator
import preprocess_utils.session2vec as sess2vec
from sklearn.preprocessing import MinMaxScaler
import utils.scaling as scale

from extract_features.global_popularity import GlobalPopularity

## ======= Datasets - Base class ======= ##

class Dataset(object):
    """ Base class containing all info about a dataset """

    def __init__(self, dataset_path):
        # load the dataset config file
        data = datasetconfig.load_config(dataset_path)
        if data is None:
            return

        self.dataset_path = dataset_path
        self.mode = data['mode']
        self.cluster = data['cluster']
        # file names
        self.train_name = data['train_name']
        self.Xtrain_name = data['Xtrain_name']
        self.Ytrain_name = data['Ytrain_name']
        self.Xtest_name = data['Xtest_name']
        # data shapes
        self.train_len = data['train_len']
        self.test_len = data['test_len']
        self.rows_per_sample = data['rows_per_sample']
        # sparsity info
        self.X_sparse_columns = data['X_sparse_columns']
        self.Y_sparse_columns = data['Y_sparse_columns']
    
        # paths
        self.train_path = os.path.join(self.dataset_path, self.train_name)
        self.X_train_path = os.path.join(self.dataset_path, self.Xtrain_name)
        self.Y_train_path = os.path.join(self.dataset_path, self.Ytrain_name)
        self.X_test_path = os.path.join(self.dataset_path, self.Xtest_name)

        # cache
        self._xtrain = None
        self._ytrain = None
        self._xtest = None
        self._xtestindices = None
    
    # load data

    def load_Xtrain(self):
        """ Load the entire X_train dataframe """
        if self._xtrain is None:
            self._xtrain = pd.read_csv(self.X_train_path).values
        return self._xtrain

    def load_Ytrain(self):
        """ Load the entire Y_train dataframe """
        if self._ytrain is None:
            self._ytrain = pd.read_csv(self.Y_train_path).values
        return self._ytrain

    def load_Xtest(self):
        """ Load the entire X_test dataframe """
        if self._xtest is None:
            self._xtest = pd.read_csv(self.X_test_path).values
        return self._xtest

    @abstractmethod
    def get_train_validation_generator(self, samples_per_batch='auto', validation_percentage=0.15):
        """ Return 2 generators (train and validation).
        samples_per_batch (int or 'auto'): number of samples to load for each batch, 'auto' will choose based on the available ram
        validation_percentage (float): percentage of samples to use for validation
        """
        pass
    
    @abstractmethod
    def get_test_generator(self, samples_per_batch='auto'):
        """ Return 2 generators (train and validation).
        samples_per_batch (int or 'auto'): number of samples to load for each batch, 'auto' will choose based on the available ram
        validation_percentage (float): percentage of samples to use for validation
        """
        pass



## ======= Datasets ======= ##

class SequenceDatasetForRegression(Dataset):

    def _preprocess_x_df(self, X_df, partial, fillNaN=0, return_indices=False):
        """ Preprocess the loaded data (X)
        partial (bool): True if X_df is a chunk of the entire file
        return_indices (bool): True to return the indices of the rows (useful at prediction time)
        """
        X_df = X_df.fillna(fillNaN)

        # add day of year column
        X_df.timestamp = pd.to_datetime(X_df.timestamp, unit='s')
        X_df['dayofyear'] = X_df.timestamp.dt.dayofyear

        cols_to_drop_in_X = ['user_id','session_id','timestamp','step','platform','city','current_filters']
        
        # scale the dataframe
        if partial:
            X_df.dayofyear /= 365
            X_df.impression_price /= 3000
        else:
            scaler = MinMaxScaler()
            X_df.loc[:,~X_df.columns.isin(cols_to_drop_in_X)] = scaler.fit_transform(
                X_df.drop(cols_to_drop_in_X, axis=1).values)

        return sess2vec.sessions2tensor(X_df, drop_cols=cols_to_drop_in_X, return_index=return_indices)

    def _preprocess_y_df(self, Y_df, fillNaN=0):
        """ Preprocess the loaded data (Y) """
        Y_df = Y_df.fillna(fillNaN)
        cols_to_drop_in_Y = ['session_id','user_id','timestamp','step']
        return sess2vec.sessions2tensor(Y_df, drop_cols=cols_to_drop_in_Y)

    # def load_train(self):
    #     train_df = pd.read_csv(self.train_path, index_col=0)
    #     train_df = train_df.fillna(0)
        # add day of year column
        # train_df.datetime = pd.to_datetime(train_df.timestamp, unit='s')
        # train_df['dayofyear'] = train_df.timestamp.dt.dayofyear
        # scale
        # scaler = MinMaxScaler()
        # train_df.impression_price = scaler.fit_transform(train_df.impression_price.values.reshape(-1,1))

        # print('train_vec:', train_df.shape)
        # return train_df

    def load_Xtrain(self):
        """ Load the entire X_train dataframe """
        if self._xtrain is None:
            self._xtrain = pd.read_csv(self.X_train_path, index_col=0)
            self._xtrain = self._preprocess_x_df(self._xtrain, partial=False)
            print('X_train:', self._xtrain.shape)
        return self._xtrain

    def load_Ytrain(self):
        """ Load the entire Y_train dataframe """
        if self._ytrain is None:
            self._ytrain = pd.read_csv(self.Y_train_path, index_col=0)
            self._ytrain = self._preprocess_y_df(self._ytrain)
            print('Y_train:', self._ytrain.shape)
        return self._ytrain

    def load_Xtest(self):
        """ Load the entire X_test dataframe """
        if self._xtest is None:
            self._xtest = pd.read_csv(self.X_test_path, index_col=0)
            self._xtest, self._xtestindices = self._preprocess_x_df(self._xtest, partial=False, return_indices=True)
            print('X_test:', self._xtest.shape)
        return self._xtest, self._xtestindices

    def get_train_validation_generator(self, validation_percentage=0.15, sessions_per_batch=256, class_weights=[]):
        # return the generator for the train and optionally the one for validation (set to 0 to skip validation)
        # if sessions_per_batch == 'auto':
        #     sessions_per_batch = self._get_auto_samples_per_batch()
        
        def prefit(Xchunk_df, Ychunk_df, index):
            """ Preprocess a chunk of the sequence dataset """
            #Xchunk_df = self._preprocess_x_df(Xchunk_df, partial=True)
            #Ychunk_df = self._preprocess_y_df(Ychunk_df)
            
            if len(class_weights) > 0:
                # weight only the last interaction (clickout item) by the class_weight
                weights = np.zeros(Xchunk_df.shape[:2])
                weights[:,-1] = Ychunk_df[:,-1,:] @ class_weights
                return Xchunk_df, Ychunk_df, weights
            else:
                return Xchunk_df, Ychunk_df

        tot_sessions = int(self.train_len / self.rows_per_sample)
        #tot_batches = math.ceil(tot_sessions / sessions_per_batch)
        
        number_of_validation_sessions = int(tot_sessions * validation_percentage)
        number_of_train_sessions = tot_sessions - number_of_validation_sessions
        train_rows = number_of_train_sessions * self.rows_per_sample

        #batches_in_train = math.ceil(number_of_train_sessions / sessions_per_batch)
        #batches_in_val = tot_batches - batches_in_train

        print('Train generator:')
        train_gen = DataGenerator(self, pre_fit_fn=prefit, rows_to_read=train_rows)
        #train_gen.name = 'train_gen'
        print('Validation generator:')
        val_gen = DataGenerator(self, pre_fit_fn=prefit, skip_rows=train_rows)
        #val_gen.name = 'val_gen'

        return train_gen, val_gen


    def get_test_generator(self, sessions_per_batch=256):
        # return the generator for the test

        #def prefit(Xchunk_df, index):
        """ Preprocess a chunk of the sequence dataset """
        #Xchunk_df = self._preprocess_x_df(Xchunk_df, partial=True)
        #return Xchunk_df

        return DataGenerator(self, for_train=False) #, pre_fit_fn=prefit)
    
    

class SequenceDatasetForClassification(Dataset):

    def _preprocess_x_df(self, X_df, partial, fillNaN=0, return_indices=False):
        """ Preprocess the loaded data (X)
        partial (bool): True if X_df is a chunk of the entire file
        return_indices (bool): True to return the indices of the rows (useful at prediction time)
        """
        X_df = X_df.fillna(fillNaN)

        # add day of year column
        #X_df.timestamp = pd.to_datetime(X_df.timestamp, unit='s')
        #X_df['dayofyear'] = X_df.timestamp.dt.dayofyear

        cols_to_drop_in_X = ['user_id','session_id','timestamp','reference','step','platform','city','current_filters']
        
        # scale the dataframe
        glo_pop_feat = GlobalPopularity()
        max_pop = glo_pop_feat.read_feature().glob_popularity.max()
        X_df.glob_popularity = scale.logarithmic(X_df.glob_popularity, max_value=max_pop)

        X_df.frequence /= 120
        # if partial:
        #     X_df.dayofyear /= 365
        #     X_df.impression_price /= 3000
        # else:
        #     scaler = MinMaxScaler()
        #     X_df.loc[:,~X_df.columns.isin(cols_to_drop_in_X)] = scaler.fit_transform(
        #         X_df.drop(cols_to_drop_in_X, axis=1).values)

        return sess2vec.sessions2tensor(X_df, drop_cols=cols_to_drop_in_X, return_index=return_indices)

    def _preprocess_y_df(self, Y_df, fillNaN=0):
        """ Preprocess the loaded data (Y) """
        Y_df = Y_df.fillna(fillNaN)
        cols_to_drop_in_Y = ['session_id','user_id','timestamp','step']
        return sess2vec.sessions2tensor(Y_df, drop_cols=cols_to_drop_in_Y)

    # def load_train(self):
    #     train_df = pd.read_csv(self.train_path, index_col=0)
    #     train_df = train_df.fillna(0)
        # add day of year column
        # train_df.datetime = pd.to_datetime(train_df.timestamp, unit='s')
        # train_df['dayofyear'] = train_df.timestamp.dt.dayofyear
        # scale
        # scaler = MinMaxScaler()
        # train_df.impression_price = scaler.fit_transform(train_df.impression_price.values.reshape(-1,1))

        # print('train_vec:', train_df.shape)
        # return train_df

    def load_Xtrain(self):
        """ Load the entire X_train dataframe """
        if self._xtrain is None:
            self._xtrain = pd.read_csv(self.X_train_path, index_col=0)
            self._xtrain = self._preprocess_x_df(self._xtrain, partial=False)
            print('X_train:', self._xtrain.shape)
        return self._xtrain

    def load_Ytrain(self):
        """ Load the entire Y_train dataframe """
        if self._ytrain is None:
            self._ytrain = pd.read_csv(self.Y_train_path, index_col=0)
            self._ytrain = self._preprocess_y_df(self._ytrain)
            print('Y_train:', self._ytrain.shape)
        return self._ytrain

    def load_Xtest(self):
        """ Load the entire X_test dataframe """
        if self._xtest is None:
            self._xtest = pd.read_csv(self.X_test_path, index_col=0)
            self._xtest, self._xtestindices = self._preprocess_x_df(self._xtest, partial=False, return_indices=True)
            print('X_test:', self._xtest.shape)
        return self._xtest, self._xtestindices

    def get_train_validation_generator(self, validation_percentage=0.15, sessions_per_batch=256, class_weights=[]):
        # return the generator for the train and optionally the one for validation (set to 0 to skip validation)
        # if sessions_per_batch == 'auto':
        #     sessions_per_batch = self._get_auto_samples_per_batch()
        
        def prefit(Xchunk_df, Ychunk_df, index):
            """ Preprocess a chunk of the sequence dataset """
            #Xchunk_df = self._preprocess_x_df(Xchunk_df, partial=True)
            #Ychunk_df = self._preprocess_y_df(Ychunk_df)
            
            if len(class_weights) > 0:
                # weight only the last interaction (clickout item) by the class_weight
                weights = np.zeros(Xchunk_df.shape[:2])
                weights[:,-1] = Ychunk_df[:,-1,:] @ class_weights
                return Xchunk_df, Ychunk_df, weights
            else:
                return Xchunk_df, Ychunk_df

        tot_sessions = int(self.train_len / self.rows_per_sample)
        #tot_batches = math.ceil(tot_sessions / sessions_per_batch)
        
        number_of_validation_sessions = int(tot_sessions * validation_percentage)
        number_of_train_sessions = tot_sessions - number_of_validation_sessions
        train_rows = number_of_train_sessions * self.rows_per_sample

        #batches_in_train = math.ceil(number_of_train_sessions / sessions_per_batch)
        #batches_in_val = tot_batches - batches_in_train

        print('Train generator:')
        train_gen = DataGenerator(self, pre_fit_fn=prefit, rows_to_read=train_rows)
        #train_gen.name = 'train_gen'
        print('Validation generator:')
        val_gen = DataGenerator(self, pre_fit_fn=prefit, skip_rows=train_rows)
        #val_gen.name = 'val_gen'

        return train_gen, val_gen


    def get_test_generator(self, sessions_per_batch=256):
        # return the generator for the test
        
        def prefit(Xchunk_df, index):
            """ Preprocess a chunk of the sequence dataset """
            #Xchunk_df = self._preprocess_x_df(Xchunk_df, partial=True)
            return Xchunk_df

        return DataGenerator(self, for_train=False, pre_fit_fn=prefit)
