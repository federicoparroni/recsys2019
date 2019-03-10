import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
import utils.get_action_score as gas
import data

def urm(train_df, test_df, save=True, clickout_score=5, impressions_score=1):
  # Return a sparse matrix (sessions, accomodations) and the association dict sessionId-urm_row
  accomodations_array = data.accomodations_id()
  assert clickout_score > impressions_score
  
  handle = create_handle(test_df, save=save)

  train_df = train_df[train_df['action_type'] == 'clickout item'].fillna(-1)
  test_df = test_df[test_df['action_type'] == 'clickout item'].fillna(-1)

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




def urm_session_aware(train_df, test_df, time_weight, save=True):
  global tw
  tw = time_weight

  accomodations_array = data.accomodations_id()
  handle = create_handle(test_df, save=save)

  # fill missing clickout_item on the test dataframe
  test_df.fillna({'reference': -1}, inplace=True)

  # concatenate the train df and the test df mantaining only the columns of interest
  df = pd.concat([train_df, test_df])[['session_id', 'action_type', 'reference', 'impressions']]

  session_groups = df.groupby('session_id')
  session_ids = list(session_groups.groups.keys())

  rows_count = len(session_groups)
  cols_count = len(accomodations_array)

  # create dictionary (k: sessionId - v: urm row)
  row_of_sessionid = {}
  for i in range(len(session_ids)):
    row_of_sessionid[session_ids[i]] = i

  # create dictionary (k: accomodationId - v: urm col)
  col_of_accomodation = {}
  for i in range(cols_count):
    col_of_accomodation[accomodations_array[i]] = i

  print("dictionaries created\n")

  sessions_score = session_groups.apply(_compute_session_score).values

  # TODO: Can be optimized using data indptr and indeces
  urm = sps.csr_matrix((rows_count, cols_count), dtype=np.float)

  for i in range(rows_count):
    score_dict = sessions_score[i]
    for k in score_dict.keys():
      col_indx = col_of_accomodation[k]
      urm[i ,col_indx] = score_dict[k]

  print("URM created\n")

  if save == True:
    if not os.path.exists('dataset/matrices'):
      os.mkdir('dataset/matrices')
    sps.save_npz('dataset/matrices/urm.npz', urm)
    print("URM saved")

    np.save('dataset/matrices/dict_row.npy', row_of_sessionid)
    np.save('dataset/matrices/dict_col.npy', col_of_accomodation)
    print("dictionaries saved")



def _compute_session_score(df):
  global  tw
  session_len = df.shape[0]
  #get the array of the weight based on the length
  weight_array = gas.time_weight(tw, session_len)
  scores = {}

  for i in range(session_len):
    row = df.iloc[i]
    session_action = row['action_type']
    score = gas.get_action_score(session_action)

    if not isinstance(score, int):
      continue

    # weight the score by the time
    score *= weight_array[i]

    # get the reference to which assign the score
    reference_id = row['reference']

    # was a test row in which we have to predict the clickout
    if reference_id == -1:
      continue

    #check if the reference is in the dictionary
    if reference_id not in scores.keys():
      scores[reference_id] = score
    else:
      scores[reference_id] += score

  return scores



def create_handle(test_df, save=True, name='handle.csv', folder='dataset/preprocessed'):
  # user_id,session_id,timestamp,step,reference,impressions
  df_handle = test_df[['user_id','session_id','timestamp','step','impressions']]
  df_handle = df_handle[(test_df['action_type'] == 'clickout item') & (test_df['reference'].isnull())]
  
  df_handle.to_csv('{}/{}'.format(folder,name), index=False)
  return df_handle


if __name__ == "__main__":
  train_df = pd.read_csv('dataset/preprocessed/local_train.csv')
  test_df = pd.read_csv('dataset/preprocessed/local_test.csv')
  #u, session_ids, col_of_accomodation, handle = urm(train_df, test_df, save=False)

  urm_session_aware(train_df, test_df, 'lin', save=False)