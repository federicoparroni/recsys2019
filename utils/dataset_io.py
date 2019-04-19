import os
import json
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import preprocess_utils.session2vec as sess2vec
from generator import DataGenerator

class DatasetConfig(object):
    """ Base class containing all info about a dataset """

    def __init__(self, dataset_path, data):
        self.dataset_path = dataset_path
        # file names
        self.Xtrain_name = data['Xtrain_name']
        self.Ytrain_name = data['Ytrain_name']
        self.Xtest_name = data['Xtest_name']
        # data shapes
        self.train_len = data['train_len']
        self.test_len = data['test_len']
        self.output_size = data['output_size']
        self.rows_per_sample = data['rows_per_sample']
        # sparsity info
        self.X_sparse_columns = data['X_sparse_cols']
        self.Y_sparse_columns = data['Y_sparse_cols']
    
    def x_train_path(self):
        return os.path.join(self.dataset_path, self.Xtrain_name)
    
    def y_train_path(self):
        return os.path.join(self.dataset_path, self.Ytrain_name)

    def x_test_path(self):
        return os.path.join(self.dataset_path, self.Xtest_name)
    
    def load_Xtrain(self):
        return pd.read_csv(self.x_train_path)

    def load_Ytrain(self):
        return pd.read_csv(self.y_train_path)

    def load_Xtest(self):
        return pd.read_csv(self.x_test_path)

    def get_train_validation_generator(self, validation_percentage=0.15):
        pass
    
    def get_test_generator(self):
        pass


def save_config(dataset_path, train_len, test_len, output_size, Xtrain_name='X_train.csv', Ytrain_name='Y_train.csv',
                Xtest_name='X_test.csv', rows_per_sample=1, X_sparse_cols=[], Y_sparse_cols=[]):
    """ Save the config file for the specified dataset """
    path = os.path.join(dataset_path, 'dataset_config.json')
    data = {
        'Xtrain_name': Xtrain_name,
        'Ytrain_name': Ytrain_name,
        'train_len': train_len,
        'Xtest_name': Xtest_name,
        'test_len': test_len,
        'output_size': output_size,
        'rows_per_sample': rows_per_sample,
        'X_sparse_columns': X_sparse_cols,
        'Y_sparse_columns': Y_sparse_cols,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
def load_config(dataset_path, return_config_obj=True):
    """ Read the config file for the specified dataset and return the config object """
    path = os.path.join(dataset_path, 'dataset_config.json')
    with open(path, 'r') as f:
        data = json.load(f)
        return DatasetConfig(dataset_path, data) if return_config_obj else data


## ======= Datasets ======= ##

class SequenceDataset(DatasetConfig):

    def _preprocess_x_df(self, X_df):
        X_df.timestamp = pd.to_datetime(X_df.timestamp, unit='s')
        X_df['dayofyear'] = X_df.timestamp.dt.dayofyear

        cols_to_drop_in_X = ['user_id','session_id','timestamp','step','platform','city','current_filters']
        
        scaler = MinMaxScaler()
        X_df.loc[:,~X_df.columns.isin(cols_to_drop_in_X)] = scaler.fit_transform(
            X_df.drop(cols_to_drop_in_X, axis=1).values)

        return sess2vec.sessions2tensor(X_df, drop_cols=cols_to_drop_in_X)

    def _preprocess_y_df(self, Y_df):
        cols_to_drop_in_Y = ['session_id','user_id','timestamp','step']
        return sess2vec.sessions2tensor(Y_df, drop_cols=cols_to_drop_in_Y)


    def load_Xtrain(self):
        X_train_df = pd.read_csv(self.x_train_path, index_col=0)
        return self._preprocess_x_df(X_train_df)

    def load_Ytrain(self):
        Y_train_df = pd.read_csv(self.y_train_path, index_col=0)
        return self._preprocess_y_df(Y_train_df)

    def load_Xtest(self):
        X_test_df = pd.read_csv(self.x_test_path, index_col=0)
        return self._preprocess_x_df(X_test_df)

    def get_train_validation_generator(self, validation_percentage=0.15):
        # return the generator for the train and optionally the one for validation (set to 0 to skip validation)
        def prefit(Xchunk_df, Ychunk_df, index):
            """ Preprocess a chunk of the sequence dataset """
            Xchunk_df.timestamp = pd.to_datetime(Xchunk_df.timestamp, unit='s')
            Xchunk_df['dayofyear'] = Xchunk_df.timestamp.dt.dayofyear

            # scale the dataframe
            Xchunk_df.dayofyear /= 365
            Xchunk_df.impression_price /= 3000

            cols_to_drop_in_X = ['user_id','session_id','timestamp','step','platform','city','current_filters']
            cols_to_drop_in_Y = ['session_id','user_id','timestamp','step']

            X_train = sess2vec.sessions2tensor(Xchunk_df, drop_cols=cols_to_drop_in_X)
            Y_train = sess2vec.sessions2tensor(Ychunk_df, drop_cols=cols_to_drop_in_Y)
            print(index, ' - X_train:', X_train.shape, 'Y_train:', Y_train.shape)

        tot_sessions = self.train_len / self.rows_per_sample
        number_of_validation_sessions = int(self.train_len * validation_percentage)
        number_of_train_sessions = tot_sessions - number_of_validation_sessions
        validation_rows = number_of_validation_sessions * self.rows_per_sample
        
        train_gen = DataGenerator(self.dataset_path, pre_fit_fn=prefit, batches_per_epoch=number_of_train_sessions)
        val_gen = DataGenerator(self.dataset_path, pre_fit_fn=prefit, batches_per_epoch=number_of_validation_sessions,
                                skip_rows=validation_rows)

        return train_gen, val_gen


    def get_test_generator(self):
        # return the generator for the test
        def prefit(Xchunk_df, index):
            """ Preprocess a chunk of the sequence dataset """
            Xchunk_df.timestamp = pd.to_datetime(Xchunk_df.timestamp, unit='s')
            Xchunk_df['dayofyear'] = Xchunk_df.timestamp.dt.dayofyear

            # scale the dataframe
            Xchunk_df.dayofyear /= 365
            Xchunk_df.impression_price /= 3000

            cols_to_drop_in_X = ['user_id','session_id','timestamp','step','platform','city','current_filters']
            X_train = sess2vec.sessions2tensor(Xchunk_df, drop_cols=cols_to_drop_in_X)
            
            print(index, ' - X_train:', X_train.shape, 'Y_train:', Y_train.shape)

        return DataGenerator(self.dataset_path, for_train=False, pre_fit_fn=prefit)
