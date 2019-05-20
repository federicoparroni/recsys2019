import pandas as pd
from tqdm import tqdm
import data
from utils.check_folder import check_folder


def extract_features_from_full(mode, cluster='no_cluster'):

    def _set_no_reordering(seq):
        """
        Remove duplicates maintaining ordering
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    features_full = pd.read_csv('dataset/preprocessed/no_cluster/full/feature/past_session_features/features.csv')

    print('loaded full')
    features = features_full[['user_id', 'session_id']]

    full_sm = pd.concat([data.train_df(mode, cluster), data.test_df(mode, cluster)])

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
    print('Len of resulting features is '.format(len(features_target)))
    path = 'dataset/preprocessed/{}/{}/feature/past_session_features/features.csv'.format(cluster,mode)
    check_folder(path)
    features_target.to_csv(path, index=False)