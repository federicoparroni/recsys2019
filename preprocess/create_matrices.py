import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer

def urm(train_df, test_df, accomodations_array, save=True, clickout_score=5, impressions_score=1):
  # Return a sparse matrix (sessions, accomodations) and the association dict sessionId-urm_row
  assert clickout_score > impressions_score
  
  train_df = train_df[train_df['action_type'] == 'clickout item'].fillna(0)
  test_df = test_df[test_df['action_type'] == 'clickout item'].fillna(0)

  df = pd.concat([train_df, test_df])[['session_id','reference','impressions']]
  session_groups = df.groupby('session_id')

  session_ids = list(session_groups.groups.keys())

  df_references = session_groups.reference.apply(lambda x: list(map(int,x))).reset_index(name='references')

  df_impressions = session_groups.impressions.apply(lambda x: list(map(int, x.values[0].split('|')))).reset_index(name='impressions')
  
  # one hot of references and impressions
  mlb = MultiLabelBinarizer(accomodations_array, sparse_output=True)

  clickout_onehot = mlb.fit_transform(df_references.references)
  
  impr_onehot = mlb.fit_transform(df_impressions.impressions)

  urm = (clickout_score - impressions_score) * clickout_onehot + impressions_score * impr_onehot

  # create dictionary (k: sessionId - v: urm row)
  row_of_sessionid = {}
  for i in range(len(session_ids)):
    row_of_sessionid[session_ids[i]] = i

  # create dictionary (k: accomodationId - v: urm col)
  col_of_accomodation = {}
  for i in range(len(mlb.classes)):
    col_of_accomodation[mlb.classes[i]] = i
  
  if save == True:
    sps.save_npz('dataset/matrices/train_urm.npz', urm)
    np.save('dataset/matrices/dict_row.npy', row_of_sessionid)
    np.save('dataset/matrices/dict_col.npy', col_of_accomodation)
  return urm, row_of_sessionid, col_of_accomodation


if __name__ == "__main__":
  import data
  train_df = pd.read_csv('dataset/preprocessed/local_train.csv')
  test_df = pd.read_csv('dataset/preprocessed/local_test.csv')
  accomodations = data.accomodations_df()['item_id']
  u, session_ids, col_of_accomodation = urm(train_df, test_df, accomodations)

  print(u.shape)
  print(session_ids)
