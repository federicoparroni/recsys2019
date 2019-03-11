import pandas as pd
import scipy.sparse as sps
import numpy as np

__mode__ = {
  'full': 0,
  'local': 1,
  'small': 2
}
                  # full paths                      # local paths                           # small paths
TRAIN_PATH = ['dataset/original/train.csv', 'dataset/preprocessed/local/train.csv', 'dataset/preprocessed/small/train.csv']
TEST_PATH = ['dataset/original/test.csv', 'dataset/preprocessed/local/test.csv', 'dataset/preprocessed/small/test.csv']
HANDLE_PATH = ['dataset/preprocessed/full/handle.csv', 'dataset/preprocessed/local/handle.csv', 'dataset/preprocessed/small/handle.csv']
URM_PATH = ['dataset/matrices/full/', 'dataset/matrices/local/', 'dataset/matrices/small/']
DICT_ROW_PATH = ['dataset/matrices/full/dict_row.npy', 'dataset/matrices/local/dict_row.npy', 'dataset/matrices/small/dict_row.npy'] 

ITEMS_PATH = 'dataset/original/item_metadata.csv'
DICT_COL_PATH = 'dataset/matrices/dict_col.npy'


_df_train = [None, None, None]
_df_test = [None, None, None]
_df_handle = [None, None, None]
_df_items = None
_df_items_ids = None
# URM structures
_urm = [None, None, None]
_dict_row = [None, None, None]
_dict_col = None
_target_urm_rows = [None, None, None]


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

def handle_df(mode):
  idx = __mode__[mode]
  if _df_handle[idx] is None:
    _df_handle[idx] = pd.read_csv(HANDLE_PATH[idx])
  return _df_handle[idx]

def accomodations_df():
  global _df_items
  if _df_items is None:
    _df_items = pd.read_csv(ITEMS_PATH)
  return _df_items

def accomodations_ids():
  global _df_items_ids
  if _df_items_ids is None:
    _df_items_ids = accomodations_df()['item_id'].values
  return _df_items_ids

# URM structures
def urm(mode, urm_name='urm'):
  idx = __mode__[mode]
  urm_path = '{}{}.npz'.format(URM_PATH[idx], urm_name)
  if _urm[idx] is None:
    _urm[idx] = sps.load_npz(urm_path)
  return _urm[idx]

def dictionary_row(mode):
  idx = __mode__[mode]
  if _dict_row[idx] is None:
    _dict_row[idx] = np.load(DICT_ROW_PATH[idx]).item()
  return _dict_row[idx]

def dictionary_col():
  global _dict_col
  if _dict_col is None:
    _dict_col = np.load(DICT_COL_PATH).item()
  return _dict_col

def target_urm_rows(mode):
  idx = __mode__[mode]
  dictionary_row(mode)
  if _target_urm_rows[idx] is None:
    _target_urm_rows[idx] = []
    for r in handle_df(mode).session_id.values:
      _target_urm_rows[idx].append(_dict_row[r])
  return _target_urm_rows[idx]
