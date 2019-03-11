import os
import sys
sys.path.append(os.getcwd())

import data
import pandas as pd
import matplotlib.pyplot as plt

def show(df):
  df = df[['session_id', 'action_type']].groupby('session_id').agg('count')
  print(df.head(20))
  print('...')
  print(df['action_type'].max())
  df.hist(bins='auto')
  plt.show(block=True)

if __name__ == "__main__":
  #df = pd.read_csv('dataset/preprocessed/train_small.csv')
  df = data.train_df()
  show(df)