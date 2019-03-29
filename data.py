import pandas as pd
import scipy.sparse as sps
import numpy as np

# original files
TRAIN_ORIGINAL_PATH = 'dataset/original/train.csv'
TEST_ORIGINAL_PATH = 'dataset/original/test.csv'
ITEMS_ORIGINAL_PATH = 'dataset/original/item_metadata.csv'

# full df
FULL_PATH = 'dataset/preprocessed/full.csv'

URM_PATH = ['dataset/matrices/full/', 'dataset/matrices/local/', 'dataset/matrices/small/']
DICT_ROW_PATH = ['dataset/matrices/full/dict_row.npy', 'dataset/matrices/local/dict_row.npy', 'dataset/matrices/small/dict_row.npy'] 
DICT_COL_PATH = ['dataset/matrices/full/dict_col.npy', 'dataset/matrices/local/dict_col.npy', 'dataset/matrices/small/dict_col.npy']

ITEMS_PATH = 'dataset/preprocessed/item_metadata.csv'

# config file
CONFIG_FILE_PATH = 'dataset/preprocessed/config.pkl'
TRAIN_LEN_KEY = 'max_train_idx'

_df_train_original = None
_df_test_original = None
_df_original_items = None

_df_full = None

_df_train = {}
_df_test = {}
_target_indices = {}

_df_items = None
_df_items_ids = None
# URM structures
_urm = [None, None, None]
_icm = None
_dict_row = [None, None, None]
_dict_col = [None, None, None]
_target_urm_rows = [None, None, None]


def full_df():
  global _df_full
  if _df_full is None:
    _df_full = pd.read_csv(FULL_PATH, index_col=0)
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
    _df_train[path] = pd.read_csv(path, index_col=0)
    _df_train[path].drop(['index'], axis=1, inplace=True)
  return _df_train[path]

def test_df(mode, cluster='no_cluster'):
  global _df_test
  path = 'dataset/preprocessed/{}/{}/test.csv'.format(cluster, mode)
  if path not in _df_test:
    _df_test[path] = pd.read_csv(path, index_col=0)
    _df_test[path].drop(['index'], axis=1, inplace=True)
  return _df_test[path]

def target_indices(mode, cluster='no_cluster'):
  global _target_indices
  path = 'dataset/preprocessed/{}/{}/target_indices.npy'.format(cluster, mode)
  if path not in _target_indices:
    _target_indices[path] = np.load(path)
  return _target_indices[path]

def accomodations_df():
  global _df_items
  if _df_items is None:
    _df_items = pd.read_csv(ITEMS_PATH)
  return _df_items

def accomodations_ids():
  global _df_items_ids
  if _df_items_ids is None:
    _df_items_ids = list(map(int, accomodations_original_df()['item_id'].values))
  return _df_items_ids

def accomodations_original_df():
  global _df_original_items
  if _df_original_items is None:
    _df_original_items = pd.read_csv(ITEMS_ORIGINAL_PATH)
  return _df_original_items

# URM structures
def urm(mode, urm_name='urm_clickout'):
  idx = __mode__[mode]
  urm_path = '{}{}.npz'.format(URM_PATH[idx], urm_name)
  if _urm[idx] is None:
    _urm[idx] = sps.load_npz(urm_path)
  return _urm[idx]

def icm():
  global _icm
  # note is used the urm path since it is dataset/matrices/full/
  icm_path = '{}{}.npz'.format(URM_PATH[0], 'icm')
  if _icm is None:
    _icm = sps.load_npz(icm_path)
  return _icm

def dictionary_row(mode):
  idx = __mode__[mode]
  if _dict_row[idx] is None:
    _dict_row[idx] = np.load(DICT_ROW_PATH[idx]).item()
  return _dict_row[idx]

def dictionary_col(mode):
  # global _dict_col
  idx = __mode__[mode]
  if _dict_col[idx] is None:
    _dict_col[idx] = np.load(DICT_COL_PATH[idx]).item()
  return _dict_col[idx]

def target_urm_rows(mode):
  idx = __mode__[mode]
  dictionary_row(mode)
  if _target_urm_rows[idx] is None:
    _target_urm_rows[idx] = []
    for r in handle_df(mode).session_id.values:
      _target_urm_rows[idx].append(_dict_row[idx][r])
  return _target_urm_rows[idx]
