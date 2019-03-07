import pandas as pd

TRAIN_CSV_PATH = 'dataset/original/train.csv'
TEST_CSV_PATH = 'dataset/original/test.csv'

_cache_df_train = None
_cache_df_test = None

def df_train():
  global _cache_df_train
  if _cache_df_train is None:
    _cache_df_train = pd.read_csv(TRAIN_CSV_PATH)
  return _cache_df_train

def df_test():
  global _cache_df_test
  if _cache_df_train is None:
    _cache_df_test = pd.read_csv(pd.read_csv(TEST_CSV_PATH))
  return _cache_df_test
