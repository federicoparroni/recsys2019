import os
import json


class DatasetConfig(object):

    def __init__(self, dataset_path, data):
        self.dataset_path = dataset_path
        self.Xtrain_name = data['Xtrain_name']
        self.Ytrain_name = data['Ytrain_name']
        self.train_len = data['train_len']
        self.Xtest_name = data['Xtest_name']
        self.test_len = data['test_len']
        self.rows_per_sample = data['rows_per_sample']
        self.X_sparse_columns = data['X_sparse_cols']
        self.Y_sparse_columns = data['Y_sparse_cols']
    
    def x_train_path(self):
        return os.path.join(self.dataset_path, self.Xtrain_name)
    
    def y_train_path(self):
        return os.path.join(self.dataset_path, self.Ytrain_name)

    def x_test_path(self):
        return os.path.join(self.dataset_path, self.Xtest_name)


def save_config(dataset_path, train_len, test_len, Xtrain_name='X_train.csv', Ytrain_name='Y_train.csv',
                Xtest_name='X_test.csv', rows_per_sample=1, X_sparse_cols=[], Y_sparse_cols=[]):
    """ Save the config file for the specified dataset """
    path = os.path.join(dataset_path, 'dataset_config.json')
    data = {
        'Xtrain_name': Xtrain_name,
        'Ytrain_name': Ytrain_name,
        'train_len': train_len,
        'Xtest_name': Xtest_name,
        'test_len': test_len,
        'rows_per_sample': rows_per_sample,
        'X_sparse_columns': X_sparse_cols,
        'Y_sparse_columns': Y_sparse_cols,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
def load_config(dataset_path):
    """ Read the config file for the specified dataset and return the config object """
    path = os.path.join(dataset_path, 'dataset_config.json')
    with open(path, 'r') as f:
        data = json.load(f)
        return DatasetConfig(dataset_path, data)