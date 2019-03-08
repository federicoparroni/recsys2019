import pandas as pd
import scipy.sparse as sps
import numpy as np

TRAIN_CSV_PATH = 'dataset/preprocessed/local_train.csv'
TEST_CSV_PATH = 'dataset/preprocessed/local_test.csv'
HANDLE_CSV_PATH = 'dataset/preprocessed/local_handle.csv'
ITEMS_CSV_PATH = 'dataset/original/item_metadata.csv'
URM_PATH = 'dataset/matrices/urm.npz'
URM_TRAIN_PATH = 'dataset/matrices/train_urm.npz'
DICT_ROW_PATH = 'dataset/matrices/dict_row.npy'
DICT_COL_PATH = 'dataset/matrices/dict_col.npy'

_cache_df_train = None
_cache_df_test = None
_cache_df_handle = None
_cache_urm = None
_cache_df_items = None
_cache_urm_train = None
_cache_dict_row = None
_cache_dict_col = None

def train_df():
  global _cache_df_train
  if _cache_df_train is None:
    _cache_df_train = pd.read_csv(TRAIN_CSV_PATH)
  return _cache_df_train

def test_df():
  global _cache_df_test
  if _cache_df_train is None:
    _cache_df_test = pd.read_csv(TEST_CSV_PATH)
  return _cache_df_test

def urm():
  global _cache_urm
  if _cache_urm is None:
    _cache_urm = np.load(URM_PATH).item()
  return _cache_urm

def handle_df():
  global _cache_df_handle
  if _cache_df_handle is None:
    _cache_df_handle = pd.read_csv(HANDLE_CSV_PATH)
  return _cache_df_handle

def accomodations_df():
  global _cache_df_items
  if _cache_df_items is None:
    _cache_df_items = pd.read_csv(ITEMS_CSV_PATH)
  return _cache_df_items

def train_urm():
  global _cache_urm_train
  if _cache_urm_train is None:
    _cache_urm_train = sps.load_npz(URM_TRAIN_PATH)
  return _cache_urm_train

def dictionary_row():
  global _cache_dict_row
  if _cache_dict_row is None:
    _cache_dict_row = np.load(DICT_ROW_PATH).item()
  return _cache_dict_row

def dictionary_col():
  global _cache_dict_col
  if _cache_dict_col is None:
    _cache_dict_col = np.load(DICT_COL_PATH).item()
  return _cache_dict_col

  
  