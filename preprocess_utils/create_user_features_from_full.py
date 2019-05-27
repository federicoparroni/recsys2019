import pandas as pd
from tqdm import tqdm
import data
from utils.check_folder import check_folder


def extract_features_from_full(mode, cluster='no_cluster'):

    feature_name = 'past_future_session_features'
    features_full = pd.read_csv('dataset/preprocessed/no_cluster/full/feature/{}/features.csv'.format(feature_name))

    print('loaded full')

    features = features_full[['user_id', 'session_id']]

    train_sm = data.train_df(mode, cluster)
    test_sm = data.test_df(mode, cluster)
    full_sm = pd.concat([train_sm, test_sm])

    target_sessions = []
    for name, group in full_sm.groupby(['user_id', 'session_id']):
        if 'clickout item' in list(set(group.action_type.values)):
            target_sessions += [group.session_id.values[0]]

    # Avoid terminating with index out of range
    target_sessions += [-1]
    print('sorted')

    relevant_idx = []

    grouped = features.groupby(['user_id', 'session_id'])

    for name, group in tqdm(grouped):
        if group.session_id.values[0] == target_sessions[0]:
            relevant_idx += list(group.index.values)
            target_sessions.pop(0)

    features_target = features_full.iloc[relevant_idx, :]
    print('Len of resulting features is:')
    print(len(features_target))

    print('Correcting feature - delete same name sessions')

    feat_label = pd.read_csv('dataset/preprocessed/{}/{}/feature/impression_label/features.csv'.format(cluster, mode))
    print(len(feat_label))

    train_sessions = set(data.train_df('full', 'no_cluster').session_id.values)
    test_sessions = set(data.test_df('full', 'no_cluster').session_id.values)

    # Get intersection
    sessions_to_correct = list(train_sessions.intersection(test_sessions))

    print(sessions_to_correct)

    to_add = feat_label[feat_label.session_id.isin(sessions_to_correct)]
    to_add = to_add.drop(['label'], axis=1)
    cols = features_full.columns.values[3:]
    to_add[cols] = pd.DataFrame([[0]*2 + ['no_action'] + [0] * 16 + ['no_action'] + [0] * 14 ], index=to_add.index)

    features_target = features_target[~features_target.session_id.isin(sessions_to_correct)]
    features_target = pd.concat([features_target, to_add])
    features_target = features_target.sort_values(by=['user_id', 'session_id'])

    print(len(features_target))

    path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(cluster, mode, feature_name)
    check_folder(path)

    features_target.to_csv(path, index=False)
