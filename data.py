import pandas as pd
import scipy.sparse as sps
import numpy as np
import pickle
import os
import dask.dataframe as ddf

# original files
TRAIN_ORIGINAL_PATH = 'dataset/original/train.csv'
TEST_ORIGINAL_PATH = 'dataset/original/test.csv'
ITEMS_ORIGINAL_PATH = 'dataset/original/item_metadata.csv'

# full df
FULL_PATH = 'dataset/preprocessed/full.csv'

URM_PATH = ['dataset/matrices/full/', 'dataset/matrices/local/', 'dataset/matrices/small/']
DICT_ROW_PATH = ['dataset/matrices/full/dict_row.npy', 'dataset/matrices/local/dict_row.npy',
                 'dataset/matrices/small/dict_row.npy']
DICT_COL_PATH = ['dataset/matrices/full/dict_col.npy', 'dataset/matrices/local/dict_col.npy',
                 'dataset/matrices/small/dict_col.npy']

ITEMS_PATH = 'dataset/preprocessed/item_metadata.csv'
ACCOMODATIONS_1HOT_PATH = 'dataset/preprocessed/accomodations_1hot.csv'

# config file
CONFIG_FILE_PATH = 'dataset/config.pkl'
TRAIN_LEN_KEY = 'max_train_idx'

_df_train_original = None
_df_test_original = None
_df_original_items = None

_df_full = None

_df_train = {}
_df_test = {}
_target_indices = {}

_df_classification_train = {}
_df_classification_test = {}

_xgboost_train_df = {}
_xgboost_test_df = {}

_df_accomodations_one_hot = None

_user_prop = {}

_df_items = None
_df_items_ids = None
# URM structures
_urm = {}
_dict_row = {}
_dict_col = {}
_icm = None

_full_train_indices = None
_full_test_indices = None

# constants
SPLIT_USED = 'no_cluster'


def full_df():
    global _df_full
    if _df_full is None:
        print('caching df_full...')
        _df_full = pd.read_csv(FULL_PATH, index_col=0)
        print('Done!')
    return _df_full


def original_train_df():
    global _df_train_original
    if _df_train_original is None:
        _df_train_original = pd.read_csv(TRAIN_ORIGINAL_PATH)
    return _df_train_original


def original_test_df():
    global _df_test_original
    if _df_test_original is None:
        _df_test_original = pd.read_csv(TEST_ORIGINAL_PATH)
    return _df_test_original


def train_df(mode, cluster='no_cluster'):
    global _df_train
    path = 'dataset/preprocessed/{}/{}/train.csv'.format(cluster, mode)
    if path not in _df_train:
        if mode == "full" and cluster == 'no_cluster':
            print("Loading {} train_df, it will take a while..".format(mode))
        _df_train[path] = pd.read_csv(path, index_col=0)

    return _df_train[path]


def test_df(mode, cluster='no_cluster'):
    global _df_test
    path = 'dataset/preprocessed/{}/{}/test.csv'.format(cluster, mode)
    if path not in _df_test:
        _df_test[path] = pd.read_csv(path, index_col=0)
    return _df_test[path]


def target_indices(mode, cluster='no_cluster'):
    global _target_indices
    path = 'dataset/preprocessed/{}/{}/target_indices.npy'.format(cluster, mode)
    if path not in _target_indices:
        _target_indices[path] = np.load(path)
    return _target_indices[path]

def xgboost_train_df(mode, sparse=True, cluster='no_cluster'):
    global _xgboost_train_df
    path = 'dataset/preprocessed/{}/{}/xgboost/classification_train.csv'.format(cluster, mode)
    if sparse:
        tot_path = path + 'sparse'
    else:
        tot_path = path + 'dense'

    if tot_path not in _xgboost_train_df:
        if sparse:
            data = ddf.read_csv(path)
            data = data.map_partitions(lambda part: part.to_sparse(fill_value=0))
            data = data.compute().reset_index(drop=True)
            data = data.drop(['Unnamed: 0'], axis=1)
            _xgboost_train_df[tot_path] = data
        else:
            _xgboost_train_df[tot_path] = pd.read_csv(path, index_col=0)

    return _xgboost_train_df[tot_path]

def xgboost_test_df(mode, sparse=True, cluster='no_cluster'):
    global _xgboost_test_df
    path = 'dataset/preprocessed/{}/{}/xgboost/classification_test.csv'.format(cluster, mode)
    if sparse:
        tot_path = path + 'sparse'
    else:
        tot_path = path + 'dense'

    if tot_path not in _xgboost_train_df:
        if sparse:
            data = ddf.read_csv(path)
            data = data.map_partitions(lambda part: part.to_sparse(fill_value=0))
            data = data.compute().reset_index(drop=True)
            data = data.drop(['Unnamed: 0'], axis=1)
            _xgboost_test_df[tot_path] = data
        else:
            _xgboost_test_df[tot_path] = pd.read_csv(path, index_col=0)

    return _xgboost_test_df[tot_path]

def classification_train_df(mode, sparse=True, cluster='no_cluster', algo='xgboost'):
    global _df_classification_train
    path = 'dataset/preprocessed/{}/{}/classification_train_{}.csv'.format(cluster, mode, algo)
    if sparse:
        tot_path = path + 'dense'
    else:
        tot_path = path + 'sparse'

    if tot_path not in _df_classification_train:
        if sparse:
            data = ddf.read_csv(path)
            data = data.map_partitions(lambda part: part.to_sparse(fill_value=0))
            data = data.compute().reset_index(drop=True)
            data = data.drop(['Unnamed: 0'], axis=1)
            _df_classification_train[tot_path] = data
        else:
            _df_classification_train[tot_path] = pd.read_csv(path, index_col=0)

    return _df_classification_train[tot_path]


def classification_test_df(mode, sparse=True, cluster='no_cluster', algo='xgboost'):
    global _df_classification_test
    path = 'dataset/preprocessed/{}/{}/classification_test_{}.csv'.format(cluster, mode, algo)
    if sparse:
        tot_path = path + 'dense'
    else:
        tot_path = path + 'sparse'

    if tot_path not in _df_classification_test:
        if sparse:
            data = ddf.read_csv(path, dtype={'1 Night filter active when clickout': 'float64',
                                             '1 Star filter active when clickout': 'float64',
                                             '2 Nights filter active when clickout': 'float64',
                                             '2 Star filter active when clickout': 'float64',
                                             '3 Star filter active when clickout': 'float64',
                                             '4 Star filter active when clickout': 'float64',
                                             '5 Star filter active when clickout': 'float64',
                                             'Accessible Hotel filter active when clickout': 'float64',
                                             'interaction_item_deals_session_ref_not_in_impr': 'float64',
                                             'interaction_item_deals_session_ref_this_impr': 'float64',
                                             'interaction_item_image_session_ref_not_in_impr': 'float64',
                                             'interaction_item_rating_session_ref_not_in_impr': 'float64'})
            data = data.map_partitions(
                lambda part: part.to_sparse(fill_value=0))
            data = data.compute().reset_index(drop=True)
            data = data.drop(['Unnamed: 0'], axis=1)
            _df_classification_test[tot_path] = data
        else:
            _df_classification_test[tot_path] = pd.read_csv(path, index_col=0)

    return _df_classification_test[tot_path]


def user_prop_df(mode, sparse=True, cluster='no_cluster'):
    global _user_prop
    path = 'dataset/preprocessed/{}/{}/user_properties.csv'.format(cluster, mode)

    if sparse:
        tot_path = path + 'dense'
    else:
        tot_path = path + 'sparse'

    if tot_path not in _user_prop:
        if sparse:
            data = ddf.read_csv(path)
            data = data.map_partitions(lambda part: part.to_sparse(fill_value=0))
            data = data.compute().reset_index(drop=True)
            data = data.drop(['Unnamed: 0'], axis=1)
            _user_prop[tot_path] = data
        else:
            _user_prop[tot_path] = pd.read_csv(path, index_col=0)

    return _user_prop[tot_path]


def train_indices(mode):
    global _full_train_indices
    path = 'dataset/preprocessed/{}/{}/train_indices.npy'.format(SPLIT_USED, mode)
    if _full_train_indices is None:
        _full_train_indices = pd.Index(np.load(path))
    return _full_train_indices


def test_indices(mode):
    global _full_test_indices
    path = 'dataset/preprocessed/{}/{}/test_indices.npy'.format(SPLIT_USED, mode)
    if _full_test_indices is None:
        _full_test_indices = pd.Index(np.load(path))
    return _full_test_indices


def accomodations_df():
    global _df_items
    if _df_items is None:
        _df_items = pd.read_csv(ITEMS_PATH)
    return _df_items


def accomodations_ids():
    global _df_items_ids
    if _df_items_ids is None:
        _df_items_ids = list(map(int, accomodations_df()['item_id'].values))
    return _df_items_ids


def accomodations_original_df():
    global _df_original_items
    if _df_original_items is None:
        _df_original_items = pd.read_csv(ITEMS_ORIGINAL_PATH)
    return _df_original_items


def accomodations_one_hot():
    global _df_accomodations_one_hot
    if not os.path.isfile(ACCOMODATIONS_1HOT_PATH):
        print('Accomodations one-hot not found! Creating it...')
        import preprocess_utils.session2vec as sess2vec
        sess2vec.save_accomodations_one_hot(accomodations_original_df(), ACCOMODATIONS_1HOT_PATH)
    if _df_accomodations_one_hot is None:
        print('Loading accomodations one-hot...')
        _df_accomodations_one_hot = pd.read_csv(ACCOMODATIONS_1HOT_PATH, index_col=0).astype('Int8')
    return _df_accomodations_one_hot


# URM structures
def urm(mode, cluster, type, urm_name='urm_clickout'):
    global _urm
    path = f'dataset/preprocessed/{cluster}/{mode}/matrices/{type}/{urm_name}.npz'
    if path not in _urm:
        _urm[path] = sps.load_npz(path)
    return _urm[path]


def icm():
    global _icm
    # note is used the urm path since it is dataset/matrices/full/
    icm_path = '{}{}.npz'.format(URM_PATH[0], 'icm')
    if _icm is None:
        _icm = sps.load_npz(icm_path)
    return _icm


def dictionary_row(mode, urm_name, type, cluster='no_cluster'):
    global _dict_row
    path = f'dataset/preprocessed/{cluster}/{mode}/matrices/{type}/{urm_name}_dict_row.npy'
    if path not in _dict_row:
        _dict_row[path] = np.load(path).item()
    return _dict_row[path]


def dictionary_col(mode, urm_name, type, cluster='no_cluster'):
    # global _dict_col
    global _dict_col
    path = f'dataset/preprocessed/{cluster}/{mode}/matrices/{type}/{urm_name}_dict_col.npy'
    if path not in _dict_col:
        _dict_col[path] = np.load(path).item()
    return _dict_col[path]


# those 2 functions let you save arbitrary fields in this file and recover those back
def read_config():
    conf = None
    try:
        with open(CONFIG_FILE_PATH, 'rb') as file:
            conf = pickle.load(file)
    except IOError:
        with open(CONFIG_FILE_PATH, 'wb') as file:
            conf = {TRAIN_LEN_KEY: len(original_train_df())}
            pickle.dump(conf, file)
    return conf


def save_config(key, value):
    conf = read_config()
    conf[key] = value
    with open(CONFIG_FILE_PATH, 'wb') as file:
        pickle.dump(conf, file)
