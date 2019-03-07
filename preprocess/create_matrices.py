import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def urm(train_df, test_df, accomodations_array, clickout_score=5, impressions_score=1):
  assert clickout_score > impressions_score
  # Return a sparse matrix (sessions, accomodations)
  train_df = train_df[(train_df['action_type'] == 'clickout item') & np.logical_not(train_df['reference'].isnull())]
  test_df = test_df[(test_df['action_type'] == 'clickout item') & np.logical_not(test_df['reference'].isnull())]

  df = pd.concat([train_df, test_df])[['session_id','reference','impressions']]

  session_ids = list(df.groupby('session_id').groups.keys())

  df_references = df.groupby('session_id').reference.apply(lambda x: list(map(int,x))).reset_index(name='references')

  df_impressions = df.groupby('session_id').impressions.apply(lambda x: list(map(int, x.values[0].split('|')))).reset_index(name='impressions')
  
  # one hot of references and impressions
  mlb = MultiLabelBinarizer(accomodations, sparse_output=True)

  clickout_onehot = mlb.fit_transform(df_references.references)
  
  impr_onehot = mlb.fit_transform(df_impressions.impressions)

  return (clickout_score - impressions_score) * clickout_onehot + impressions_score * impr_onehot


if __name__ == "__main__":
  train_df = pd.read_csv('dataset/preprocessed/train_small.csv')
  test_df = pd.read_csv('dataset/preprocessed/test_small.csv')
  accomodations = pd.read_csv('dataset/original/item_metadata.csv')['item_id']
  u = urm(train_df, test_df, accomodations)

  print(u.shape)
