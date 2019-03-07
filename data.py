import pandas as pd

TRAIN_CSV_PATH = 'dataset/original/train.csv'
TEST_CSV_PATH = 'dataset/original/test.csv'
ITEMS_CSV_PATH = 'dataset/original/item_metadata.csv'

_cache_df_train = None
_cache_df_test = None
_cache_df_items = None

def train_df():
  global _cache_df_train
  if _cache_df_train is None:
    _cache_df_train = pd.read_csv(TRAIN_CSV_PATH)
  return _cache_df_train

def test_df():
  global _cache_df_test
  if _cache_df_train is None:
    _cache_df_test = pd.read_csv(pd.read_csv(TEST_CSV_PATH))
  return _cache_df_test

def accomodations_df():
  global _cache_df_items
  if _cache_df_items is None:
    _cache_df_items = pd.read_csv(pd.read_csv(ITEMS_CSV_PATH))
  return _cache_df_items
