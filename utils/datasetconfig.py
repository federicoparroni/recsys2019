import os
import json

def save_config(dataset_path, mode, cluster, train_len, test_len, train_name='', Xtrain_name='X_train.csv', 
                Ytrain_name='Y_train.csv', Xtest_name='X_test.csv', rows_per_sample=1, X_sparse_cols=[], Y_sparse_cols=[]):
    """ Save the config file for the specified dataset """
    path = os.path.join(dataset_path, 'dataset_config.json')
    data = {
        'mode': mode,
        'cluster': cluster,
        'train_name': train_name,
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
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        print('Cannot load dataset config in: {}'.format(dataset_path))
        return None