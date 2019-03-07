import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def urm(train_df, test_df, accomodations_array, clickout_score=5, impressions_score=1):
  # Return a sparse matrix (sessions, accomodations) and the association dict sessionId-urm_row
  assert clickout_score > impressions_score
  
  train_df = train_df[(train_df['action_type'] == 'clickout item') & np.logical_not(train_df['reference'].isnull())]
  test_df = test_df[(test_df['action_type'] == 'clickout item') & np.logical_not(test_df['reference'].isnull())]

  df = pd.concat([train_df, test_df])[['session_id','reference','impressions']]
  session_groups = df.groupby('session_id')

  session_ids = list(session_groups.groups.keys())

  df_references = session_groups.reference.apply(lambda x: list(map(int,x))).reset_index(name='references')

  df_impressions = session_groups.impressions.apply(lambda x: list(map(int, x.values[0].split('|')))).reset_index(name='impressions')
  
  # one hot of references and impressions
  mlb = MultiLabelBinarizer(accomodations, sparse_output=True)

  clickout_onehot = mlb.fit_transform(df_references.references)
  
  impr_onehot = mlb.fit_transform(df_impressions.impressions)

  urm = (clickout_score - impressions_score) * clickout_onehot + impressions_score * impr_onehot

  # create dictionary (k: sessionId - v: urm row)
  row_of_sessionid = {}
  for i in range(len(session_ids)):
    row_of_sessionid[session_ids[i]] = i
  
  return urm, row_of_sessionid


if __name__ == "__main__":
  import data
  train_df = pd.read_csv('dataset/preprocessed/train_small.csv')
  test_df = pd.read_csv('dataset/preprocessed/test_small.csv')
  accomodations = data.accomodations_df()['item_id']
  u, session_ids = urm(train_df, test_df, accomodations)

  print(u.shape)
  print(session_ids)
