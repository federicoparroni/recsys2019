import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
import os

def urm(train_df, test_df, accomodations_array, save=True, clickout_score=5, impressions_score=1):
  # Return a sparse matrix (sessions, accomodations) and the association dict sessionId-urm_row
  assert clickout_score > impressions_score
  
  handle = handle(test_df, save=save)

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
    if not os.path.exists('dataset/matrices'):
      os.mkdir('dataset/matrices')
    sps.save_npz('dataset/matrices/urm.npz', urm)
    np.save('dataset/matrices/dict_row.npy', row_of_sessionid)
    np.save('dataset/matrices/dict_col.npy', col_of_accomodation)
  
  return urm, row_of_sessionid, col_of_accomodation, handle

def handle(test_df, save=True, name='handle.csv', folder='dataset/preprocessed'):
  # user_id,session_id,timestamp,step,reference,impressions
  df_handle = test_df[['user_id','session_id','timestamp','step','impressions']]
  df_handle = df_handle[(test_df['action_type'] == 'clickout item') & (test_df['reference'].isnull())]
  
  df_handle.to_csv('{}/{}'.format(folder,name), index=False)
  return df_handle


if __name__ == "__main__":
  import sys
  sys.path.append(os.getcwd())

  import data
  print()
  print('(1) Create matrices from LOCAL dataset')
  print('(2) Create matrices from ORIGINAL dataset')
  choice = input()[0]

  if choice == '1':
    train_df = data.local_train_df()
    test_df = data.local_test_df

  elif choice == '2':
    train_df = data.train_df()
    test_df = data.test_df()
  
  else:
    print('Invalid option')
    exit(0)

  accomodations = data.accomodations_df()['item_id']
  u, session_ids, col_of_accomodation, handle = urm(train_df, test_df, accomodations, save=True)

  print(u.shape)
  print()
  print('Sessions: {}'.format(len(session_ids)))
  print()
  print(handle)
