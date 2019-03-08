import pandas as pd
import scipy.sparse as sps
import numpy as np

LOCAL_TRAIN_PATH = 'dataset/preprocessed/local_train.csv'
LOCAL_TEST_PATH = 'dataset/preprocessed/local_test.csv'
LOCAL_HANDLE_PATH = 'dataset/preprocessed/local_handle.csv'

TRAIN_PATH = 'dataset/original/train.csv'
TEST_PATH = 'dataset/original/test.csv'
HANDLE_PATH = 'dataset/preprocessed/handle.csv'

ITEMS_PATH = 'dataset/original/item_metadata.csv'

URM_PATH = 'dataset/matrices/urm.npz'
URM_TRAIN_PATH = 'dataset/matrices/train_urm.npz'

DICT_ROW_PATH = 'dataset/matrices/dict_row.npy'
DICT_COL_PATH = 'dataset/matrices/dict_col.npy'

LOCAL_DICT_ROW_PATH = 'dataset/matrices/local_dict_row.npy'
LOCAL_DICT_COL_PATH = 'dataset/matrices/local_dict_col.npy'


_df_train = None
_df_test = None
_df_handle = None

_df_localtrain = None
_df_localtest = None
_df_localhandle = None

_df_items = None

_urm = None
_urm_train = None
_dict_row = None
_dict_col = None
_local_dict_row = None
_local_dict_col = None

def train_df():
  global _df_train
  if _df_train is None:
    _df_train = pd.read_csv(TRAIN_PATH)
  return _df_train

def test_df():
  global _df_test
  if _df_test is None:
    _df_test = pd.read_csv(TEST_PATH)
  return _df_test

def handle_df():
  global _df_handle
  if _df_handle is None:
    _df_handle = pd.read_csv(HANDLE_PATH)
  return _df_handle

# local files
def local_train_df():
  global _df_localtrain
  if _df_localtrain is None:
    _df_localtrain = pd.read_csv(LOCAL_TRAIN_PATH)
  return _df_localtrain

def local_test_df():
  global _df_localtest
  if _df_localtest is None:
    _df_localtest = pd.read_csv(LOCAL_TEST_PATH)
  return _df_localtest

def local_handle_df():
  global _df_localhandle
  if _df_localhandle is None:
    _df_localhandle = pd.read_csv(LOCAL_HANDLE_PATH)
  return _df_localhandle


def urm():
  global _urm
  if _urm is None:
    _urm = np.load(URM_PATH).item()
  return _urm

def accomodations_df():
  global _df_items
  if _df_items is None:
    _df_items = pd.read_csv(ITEMS_PATH)
  return _df_items

def train_urm():
  global _urm_train
  if _urm_train is None:
    _urm_train = sps.load_npz(URM_TRAIN_PATH)
  return _urm_train

def dictionary_row():
  global _dict_row
  if _dict_row is None:
    _dict_row = np.load(DICT_ROW_PATH).item()
  return _dict_row

def dictionary_col():
  global _dict_col
  if _dict_col is None:
    _dict_col = np.load(DICT_COL_PATH).item()
  return _dict_col

def local_dictionary_row():
  global _local_dict_row
  if _local_dict_row is None:
    _local_dict_row = np.load(LOCAL_DICT_ROW_PATH).item()
  return _local_dict_row

def local_dictionary_col():
  global _local_dict_col
  if _local_dict_col is None:
    _local_dict_col = np.load(LOCAL_DICT_COL_PATH).item()
  return _local_dict_col

  
  