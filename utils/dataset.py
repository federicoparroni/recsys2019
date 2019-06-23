import os
import math
import pandas as pd
import numpy as np
from abc import abstractmethod
import utils.datasetconfig as datasetconfig
from generator import DataGenerator
import preprocess_utils.session2vec as sess2vec
#from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_class_weight
import utils.scaling as scale
import data
from utils.menu import single_choice

## ======= Datasets - Base class ======= ##

class DatasetBase(object):
    """ Base class to load a dataset """

    def __init__(self):
        # cache
        self._xtrain = None
        self._ytrain = None
        self._xtest = None
       
    # load data
    def load_Xtrain(self):
        """ Load the X_train dataframe """
        pass

    def load_Ytrain(self):
        """ Load the Y_train dataframe """
        pass

    def load_Xtest(self):
        """ Load the X_test dataframe """
        pass

    def load_group_train(self):
        """ 
            Loads the group array in train
            an example of the format is:
            [1, 1, 1, 1, 2, 2, 2, 3, 3]
            in the case:
            first group is composed by 4 elems
            second group is composed by 3 elems
            third group is composed by 2 elems
        """
        pass

class Dataset(DatasetBase):
    """ Base class containing all info about a dataset """

    def __init__(self, dataset_path):
        super(Dataset, self).__init__()
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
        self._xtestindices = None
        self._xtrainindices = None
    
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
    
    def get_class_weights(self, num_classes):
        y = self.load_Ytrain()
        if y.shape[1] == 1:
            # binary class
            weights = compute_class_weight('balanced', np.arange(num_classes), y[:,0])
        else:
            # multiple classes one-hot encoded
            y = [np.where(r==1)[0][0] for r in y]
            weights = compute_class_weight('balanced', np.arange(num_classes), y)

        weights = dict([ (i,w) for i,w in enumerate(weights) ])
        print(weights)
        return weights
    
    def get_sample_weights(self, num_classes):
        y = self.load_Ytrain()
        if y.shape[1] == 1:
            # binary class
            weights = compute_class_weight('balanced', np.arange(num_classes), y[:,0])
            w = y.flatten().astype('float')
            w[w == 0] = weights[0]
            w[w == 1] = weights[1]
            return w
        else:
            # multiple classes one-hot encoded
            y_2 = [np.where(r==1)[0][0] for r in y]
            weights = compute_class_weight('balanced', np.arange(num_classes), y_2)
            return (y @ weights.reshape(-1,1)).flatten()



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

    def _preprocess_x_df(self, X_df, partial, fillNaN=0, return_indices=False, load_scaler=False):
        """ Preprocess the loaded data (X)
        partial (bool): True if X_df is a chunk of the entire file
        return_indices (bool): True to return the indices of the rows (useful at prediction time)
        """
        X_df = X_df.fillna(fillNaN)

        cols_to_drop_in_X = ['user_id','session_id','timestamp','reference','step','platform','city','current_filters']
        #cols_to_drop_in_X = ['timestamp','reference','step','platform','city','current_filters']

        # scale the dataframe

        # glo_click_pop_feat = GlobalClickoutPopularity(self.mode, self.cluster)
        # max_pop = glo_click_pop_feat.read_feature()['glob_clickout_popularity'].max()
        # X_df['glob_clickout_popularity'] = scale.logarithmic(X_df['glob_clickout_popularity'], max_value=max_pop)
        
        """ trick """
        impr_count_cols = ['impr_c{}'.format(i) for i in range(25)]
        X_df.loc[:, impr_count_cols] = 1 - X_df.loc[:, impr_count_cols]

        """ scaling """
        # price_cols = ['price_{}'.format(i) for i in range(25)]
        # max_of_prices = np.max(X_df.loc[:,price_cols], axis=1)
        # mask = max_of_prices > 0
        # X_df.loc[mask, price_cols + ['price']] /= max_of_prices[mask][:,None]
        
        #X_df = X_df.drop('duration', axis=1)
        #X_df['duration'] = np.log(X_df['duration'] + 1)
        
        #X_df['glob_clickout_popularity'] = np.log(X_df['glob_clickout_popularity'] + 1)
        # for j in range(25):
        #     impr_pop_i = 'impr_pop{}'.format(j)
        #     X_df[impr_pop_i] = np.log(X_df[impr_pop_i] + 1)

        X_df.frequence = np.log(X_df.frequence.values +1)

        X_df = X_df.drop(cols_to_drop_in_X, axis=1)

        """
        scaler_path = os.path.join(self.dataset_path, 'scaler.skl')
        if load_scaler:
            scaler = joblib.load(scaler_path)
            print('Scaler loaded!')
            # scale data
            X_df.loc[:,:] = scaler.transform(X_df)
        else:
            scaler = MinMaxScaler()
            X_df.loc[:,:] = scaler.fit_transform(X_df)
            # save the scaler
            joblib.dump(scaler, scaler_path)
            print('Scaler saved!')
        """

        if return_indices:
            target = np.arange(-1, len(X_df), self.rows_per_sample)[1:]
            indices = X_df.index.values[target]
            return X_df.values.reshape((-1, self.rows_per_sample, len(X_df.columns))), indices
        else:
            return X_df.values.reshape((-1, self.rows_per_sample, len(X_df.columns)))

    def _preprocess_y_df(self, Y_df, fillNaN=0):
        """ Preprocess the loaded data (Y) """
        Y_df = Y_df.fillna(fillNaN)
        cols_to_drop_in_Y = ['session_id','user_id','timestamp','step']
        #return sess2vec.sessions2tensor(Y_df, drop_cols=cols_to_drop_in_Y)
        return Y_df.drop(cols_to_drop_in_Y, axis=1).values

    def load_Xtrain(self, return_indices=False):
        """ Load the entire X_train dataframe """
        if self._xtrain is None or (return_indices and self._xtrainindices is None):
            self._xtrain = pd.read_csv(self.X_train_path, index_col=0)
            if return_indices or self._xtrainindices is None:
                self._xtrain, self._xtrainindices = self._preprocess_x_df(self._xtrain, partial=False, return_indices=True)
            else:
                self._xtrain = self._preprocess_x_df(self._xtrain, partial=False, return_indices=False)
        
        print('X_train:', self._xtrain.shape)
        if return_indices:
            return self._xtrain, self._xtrainindices
        else:
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
            self._xtest, self._xtestindices = self._preprocess_x_df(self._xtest, partial=False, return_indices=True, load_scaler=True)
            print('X_test:', self._xtest.shape)
        return self._xtest, self._xtestindices

    def prefit_xy(self, Xchunk_df, Ychunk_df, index):
        """ Preprocess a chunk of the sequence dataset """
        #Xchunk_df = self._preprocess_x_df(Xchunk_df, partial=True)
        #Ychunk_df = self._preprocess_y_df(Ychunk_df)
        
        if len(self.class_weights) > 0:
            # weight only the last interaction (clickout item) by the class_weight
            weights = np.zeros(Xchunk_df.shape[:2])
            weights[:,-1] = Ychunk_df[:,-1,:] @ self.class_weights
            return Xchunk_df, Ychunk_df, weights
        else:
            return Xchunk_df, Ychunk_df

    def prefit_x(self, Xchunk_df, index):
        """ Preprocess a chunk of the sequence dataset """
        return Xchunk_df
    
    def get_train_validation_generator(self, validation_percentage=0.15, sessions_per_batch=256, class_weights=[]):
        # return the generator for the train and optionally the one for validation (set to 0 to skip validation)
        # if sessions_per_batch == 'auto':
        #     sessions_per_batch = self._get_auto_samples_per_batch()
        self.class_weights = class_weights

        tot_sessions = int(self.train_len / self.rows_per_sample)
        #tot_batches = math.ceil(tot_sessions / sessions_per_batch)
        
        number_of_validation_sessions = int(tot_sessions * validation_percentage)
        number_of_train_sessions = tot_sessions - number_of_validation_sessions
        train_rows = number_of_train_sessions * self.rows_per_sample

        #batches_in_train = math.ceil(number_of_train_sessions / sessions_per_batch)
        #batches_in_val = tot_batches - batches_in_train

        print('Train generator:')
        train_gen = DataGenerator(self, pre_fit_fn=self.prefit_xy, rows_to_read=train_rows)
        #train_gen.name = 'train_gen'
        print('Validation generator:')
        val_gen = DataGenerator(self, pre_fit_fn=self.prefit_xy, skip_rows=train_rows)
        #val_gen.name = 'val_gen'

        return train_gen, val_gen


    def get_test_generator(self, sessions_per_batch=256):
        # return the generator for the test
        return DataGenerator(self, for_train=False, pre_fit_fn=self.prefit_x)

    def get_class_weights(self, num_classes=25):
        return super().get_class_weights(num_classes)
    
    def get_sample_weights(self, num_classes=25):
        return super().get_sample_weights(num_classes)



class SequenceDatasetForBinaryClassification(SequenceDatasetForClassification):
    
    def prefit_xy(self, Xchunk_df, Ychunk_df, index):
        """ Preprocess a chunk of the sequence dataset """
        return Xchunk_df, Ychunk_df

    def prefit_x(self, Xchunk_df, index):
        """ Preprocess a chunk of the sequence dataset """
        return Xchunk_df

    def get_class_weights(self, num_classes=2):
        return super().get_class_weights(num_classes)
    
    def get_sample_weights(self, num_classes=2):
        return super().get_sample_weights(num_classes)


class DatasetScoresClassification(Dataset):

    def _preprocess_x_df(self, X_df, fillNaN=0):
        """ Preprocess the loaded data (X) """
        #X_df.reset_index(inplace=True)
        X_df = X_df.fillna(fillNaN)

        cols_to_drop_in_X = ['user_id','session_id','timestamp','reference','step','platform','city','current_filters']

        """ trick """
        impr_count_cols = ['impr_c{}'.format(i) for i in range(25)]
        X_df.loc[:, impr_count_cols] = 1 - X_df.loc[:, impr_count_cols]

        """ scaling """
        X_df.frequence = np.log(X_df.frequence.values +1)

        X_df = X_df.drop(cols_to_drop_in_X, axis=1)

        # return dataset with indices
        return X_df.values.reshape((-1, self.rows_per_sample, len(X_df.columns)))
    
    def _preprocess_y_df(self, Y_df, fillNaN=0):
        """ Preprocess the loaded data (Y) """
        Y_df = Y_df.fillna(fillNaN)
        cols_to_drop_in_Y = ['session_id','user_id','timestamp','step']
        return Y_df.drop(cols_to_drop_in_Y, axis=1).values

    def load_Xtrain(self):
        """ Load the entire X_train dataframe """
        if self._xtrain is None:
            self._xtrain = self._preprocess_x_df(pd.read_csv(self.X_train_path))
            print('X_train:', self._xtrain.shape)

        return self._xtrain
        
    def load_Ytrain(self):
        """ Load the entire Y_train dataframe """
        if self._ytrain is None:
            self._ytrain = self._preprocess_y_df(pd.read_csv(self.Y_train_path, index_col=0))
            print('Y_train:', self._ytrain.shape)

        return self._ytrain

    def load_Xtest(self):
        """ Load the entire X_test dataframe """
        if self._xtest is None:
            self._xtest = self._preprocess_x_df(pd.read_csv(self.X_test_path))
            print('X_test:', self._xtest.shape)

        return self._xtest

    def get_class_weights(self, num_classes=25):
        return super().get_class_weights(num_classes)

    def get_sample_weights(self, num_classes=25):
        return super().get_sample_weights(num_classes)

class DatasetScoresBinaryClassification(DatasetScoresClassification):

    def get_class_weights(self, num_classes=2):
        return super().get_class_weights(num_classes)

    def get_sample_weights(self, num_classes=2):
        return super().get_sample_weights(num_classes)


class DatasetXGBoost(DatasetBase):

    def __init__(self, mode, cluster, kind):
        super(DatasetXGBoost, self).__init__()
        self.mode = mode
        self.cluster = cluster
        self.kind = kind

    def load_Xtrain(self):
        X_train, _, _, _, _ = data.dataset_xgboost_train(mode=self.mode, cluster=self.cluster, kind=self.kind)
        return X_train

    def load_Ytrain(self):
        _, y_train, _, _, _ = data.dataset_xgboost_train(mode=self.mode, cluster=self.cluster, kind=self.kind)
        return y_train

    def load_Xtest(self):
        X_test, _, _, _ = data.dataset_xgboost_test(mode=self.mode, cluster=self.cluster, kind=self.kind)
        return X_test

    def load_group_train(self):
        _, _, groups, _, _ = data.dataset_xgboost_train(mode=self.mode, cluster=self.cluster, kind=self.kind)
        g = [list(np.ones(groups[i], dtype=np.int)*i) for i in range(len(groups))]
        g = [item for sublist in g for item in sublist]
        return np.array(g)


class DatasetLightGBM(DatasetBase):

    def __init__(self, mode, cluster, dataset_name):
        super(DatasetLightGBM, self).__init__()
        self.mode = mode
        self.cluster = cluster
        self.dataset_name = dataset_name
        self._load_data()

    def _load_data(self):
        _BASE_PATH = f'dataset/preprocessed/lightGBM/{self.cluster}/{self.mode}/{self.dataset_name}'
        self._x_train = pd.read_hdf(f'{_BASE_PATH}/x_train.hdf', key='df')
        self._y_train = np.load(f'{_BASE_PATH}/y_train.npy')
        self._group_train = np.load(f'{_BASE_PATH}/groups_train.npy')
        self._x_vali = pd.read_hdf(f'{_BASE_PATH}/x_vali.hdf', key='df')

    def load_Xtrain(self):
        return self._x_train

    def load_Ytrain(self):
        return self._y_train

    def load_Xtest(self):
        return self._x_vali

    def load_group_train(self):
        groups = self._group_train
        g = [list(np.ones(groups[i], dtype=np.int)*i) for i in range(len(groups))]
        g = [item for sublist in g for item in sublist]
        return np.array(g)


class DatasetCatboost(DatasetBase):

    def __init__(self, mode, cluster):
        super(DatasetCatboost, self).__init__()
        self.mode = mode
        self.cluster = cluster

    def load_Xtrain(self):
        train_df = data.dataset_catboost_train(mode=self.mode, cluster=self.cluster)
        # Creating univoque id for each user_id / session_id pair

        train_df = train_df.assign(
            id=(train_df['user_id'] + '_' + train_df['session_id']).astype('category').cat.codes)

        train_features = train_df.drop(['user_id', 'session_id', 'item_id'], axis=1)

        return train_features

    def load_Ytrain(self):
        return None

    def load_Xtest(self):
        test_df = data.dataset_catboost_test(mode=self.mode, cluster=self.cluster)

        target_indices = data.target_indices(self.mode, self.cluster)
        sessi_target = data.test_df(self.mode, self.cluster).loc[target_indices].session_id.values

        dict_session_trg_idx = dict(zip(sessi_target, target_indices))

        test_df['id'] = test_df.apply(
            lambda row: dict_session_trg_idx.get(row.session_id), axis=1)

        print('data for test ready')

        test_feat_df = test_df.drop(['user_id', 'session_id', 'item_id'], axis=1)

        return test_feat_df


    def load_group_train(self):
        return None


class DatasetXGBoostClassifier(DatasetBase):

    def __init__(self, mode, cluster):
        super(DatasetXGBoostClassifier, self).__init__()
        self.mode = mode
        self.cluster = cluster

    def load_Xtrain(self):
        train = data.dataset_xgboost_classifier_train(mode=self.mode, cluster=self.cluster)
        train = train.drop(["user_id", "session_id", "label"], axis=1)
        return train.values.reshape((-1, len(train.columns)))

    def load_Ytrain(self):
        train = data.dataset_xgboost_classifier_train(mode=self.mode, cluster=self.cluster)
        return train["label"].values

    def load_Xtest(self):
        test = data.dataset_xgboost_classifier_test(mode=self.mode, cluster=self.cluster)
        test = test.drop(["user_id", "session_id", "label"], axis=1)
        return test.values.reshape((-1, len(test.columns)))

    def load_group_train(self):
        return None