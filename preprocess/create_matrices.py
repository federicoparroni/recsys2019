import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
import utils.get_action_score as gas
import data


def handle(test_df, local, save=True, name='handle.csv', folder='dataset/preprocessed'):
  # user_id,session_id,timestamp,step,reference,impressions
  df_handle = test_df[['user_id','session_id','timestamp','step','impressions']]
  df_handle = df_handle[(test_df['action_type'] == 'clickout item') & (test_df['reference'].isnull())]
  
  if local:
    name = 'local_{}'.format(name)
  df_handle.to_csv('{}/{}'.format(folder,name), index=False)

  return df_handle


def urm(train_df, test_df, accomodations_array, local, clickout_score=5, impressions_score=1, save=True):
  # Return a sparse matrix (sessions, accomodations) and the association dict sessionId-urm_row
  # PARAMS
  # local: operate wether using local or original dataset
  # clickout_score: score to assign to clickout items
  # impressions_score: score to assign to impressions accomodations, must be greater than clickout_score
  assert clickout_score > impressions_score
  
  hnd = handle(test_df, local, save=save)

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
    path = 'dataset/matrices'
    if not os.path.exists(path):
      os.mkdir(path)
    path += '/'
    if local:
      path += 'local_'

    # save all
    print('Saving urm matrix... ', end='\t')
    sps.save_npz('{}urm.npz'.format(path), urm)
    print('done!')

    print('Saving row dictionary... ', end='\t')
    np.save('{}dict_row.npy'.format(path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ', end='\t')
    np.save('dataset/matrices/dict_col.npy'.format(path), col_of_accomodation)
    print('done!')

  
  return urm, row_of_sessionid, col_of_accomodation, hnd


def urm_session_aware(train_df, test_df, time_weight, save=True):
  global tw
  tw = time_weight

  accomodations_array = data.accomodations_id()
  hnd = handle(test_df, True, save=save)

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
  #TODO: RETURN STATEMENT


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
    reference_id = int(row['reference'])

    # was a test row in which we have to predict the clickout
    if reference_id == -1:
      continue

    #check if the reference is in the dictionary
    if reference_id not in scores.keys():
      scores[reference_id] = score
    else:
      scores[reference_id] += score

  return scores


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
    test_df = data.local_test_df()
    local = True
  elif choice == '2':
    train_df = data.train_df()
    test_df = data.test_df()
    local = False
  else:
    print('Invalid option')
    exit(0)
  
  accomodations = data.accomodations_df()['item_id']
  u, session_ids, col_of_accomodation, handle = urm(train_df, test_df, accomodations, local, save=True)

  print('URM shape: ', u.shape)
  print()
  print('Sessions: {}'.format(len(session_ids)))
  print()
  print('All tasks completed!')

