import data
import pandas as pd
from preprocess_utils.last_clickout_indices import find
from utils.check_folder import check_folder
from sklearn.model_selection import KFold
import numpy as np
import os.path

def preprocess_cv(mode='full'):

    def save_folds(df, user_session_df, train_index, test_index, count, mode):
        u_s_train = list(user_session_df.loc[train_index]['user_session'].values)
        u_s_test = list(user_session_df.loc[test_index]['user_session'].values)

        path = 'dataset/preprocessed/{}/{}'.format('fold_' + str(count) , mode)
        check_folder(path)

        train = df[df['user_session'].isin(u_s_train)]
        train = train.drop(['user_session'], axis=1)
        train.to_csv(os.path.join(path, 'train.csv'))
        train_indices = train.index.values
        np.save(os.path.join(path, 'train_indices'), train_indices)

        test = df[df['user_session'].isin(u_s_test)]
        target_indices = sorted(find(test))
        test.at[target_indices, 'reference'] = np.nan
        test = test.drop(['user_session'], axis=1)
        test.to_csv(os.path.join(path, 'test.csv'))
        test_indices = test.index.values
        np.save(os.path.join(path, 'test_indices'), test_indices)
        np.save(os.path.join(path, 'target_indices'), target_indices)


        print(f'Train shape : {train.shape} , Test shape : {test.shape}')
        print(f'Last clickout indices : {len(target_indices)}')

    train_df = data.train_df(mode=mode, cluster='no_cluster')
    train_df['user_session'] = train_df['user_id'].values + '_' + train_df['session_id'].values

    test_df = data.test_df(mode=mode, cluster='no_cluster')
    test_df['user_session'] = test_df['user_id'].values + '_' + test_df['session_id'].values

    df = pd.concat([train_df, test_df])

    # extract user_session referring to target_indices
    target_indices = data.target_indices(mode=mode, cluster='no_cluster')
    test_target_u_s = test_df.loc[target_indices].drop_duplicates('user_session')['user_session'].to_list()
    print(f'Number of user_session in target_indices : {len(test_target_u_s)}')

    # remove those sessions from df
    df = df[~df['user_session'].isin(test_target_u_s)]

    #df['user_session'] = df['user_id'].values + '_' + df['session_id'].values
    user_session_df = df.drop_duplicates('user_session')
    user_session_df = user_session_df.reset_index(drop=True)
    print(f'Number of user_session NOT in target_indices : {user_session_df.shape[0]}')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kf.split(user_session_df)):
        print(f' train indices : {len(train_index)}, test indices : {len(test_index)}')
        save_folds(df, user_session_df, train_index, test_index, i, mode)

if __name__ == '__main__':
    preprocess_cv()
    #preprocess_cv('small')