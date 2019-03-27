import pandas as pd
import scipy.sparse as sps
import numpy as np

__mode__ = {
  'full': 0,
  'local': 1,
  'small': 2
}

FULL_PATH = 'dataset/preprocessed/full.csv'

ORIGINAL_TRAIN_PATH = 'dataset/original/train.csv'
ORIGINAL_TEST_PATH = 'dataset/original/test.csv'

                  # full paths                      # local paths                           # small paths
TRAIN_PATH = ['dataset/original/train.csv', 'dataset/preprocessed/local/train.csv', 'dataset/preprocessed/small/train.csv', ]
TEST_PATH = ['dataset/original/test.csv', 'dataset/preprocessed/local/test.csv', 'dataset/preprocessed/small/test.csv']
URM_PATH = ['dataset/matrices/full/', 'dataset/matrices/local/', 'dataset/matrices/small/']
DICT_ROW_PATH = ['dataset/matrices/full/dict_row.npy', 'dataset/matrices/local/dict_row.npy', 'dataset/matrices/small/dict_row.npy'] 
DICT_COL_PATH = ['dataset/matrices/full/dict_col.npy', 'dataset/matrices/local/dict_col.npy', 'dataset/matrices/small/dict_col.npy']

ITEMS_ORIGINAL_PATH = 'dataset/original/item_metadata.csv'
ITEMS_PATH = 'dataset/preprocessed/full/item_metadata.csv'

_df_full = None
_df_original_train = None
_df_original_test = None
_df_train = [None, None, None]
_df_test = [None, None, None]
_df_handle = [None, None, None]
_df_items = None
_df_original_items = None
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
  global _df_original_train
  if _df_original_train is None:
    _df_original_train = pd.read_csv(ORIGINAL_TRAIN_PATH, index_col=0)
  return _df_original_train

def original_test_df():
  global _df_original_test
  if _df_original_test is None:
    _df_original_test = pd.read_csv(ORIGINAL_TEST_PATH, index_col=0)
  return _df_original_test

def train_df(mode):
  idx = __mode__[mode]
  if _df_train[idx] is None:
    _df_train[idx] = pd.read_csv(TRAIN_PATH[idx])
  return _df_train[idx]

def test_df(mode):
  idx = __mode__[mode]
  if _df_test[idx] is None:
    _df_test[idx] = pd.read_csv(TEST_PATH[idx])
  return _df_test[idx]

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
