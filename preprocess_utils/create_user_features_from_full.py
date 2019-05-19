import pandas as pd
from tqdm import tqdm
import data
from preprocess_utils.last_clickout_indices import find as find_last_clickout
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
    target_idx = find_last_clickout(full_sm)

    train_df = data.train_df('full', 'no_cluster').sort_values(by=['user_id', 'session_id'])

    print('sorted')
    target_sessions = _set_no_reordering(list(train_df.iloc[target_idx, :].sort_values(by=['user_id', 'session_id']).session_id.values))

    relevant_idx = []

    grouped = features.groupby(['user_id', 'session_id'])

    for name, group in tqdm(grouped):
        print(group.session_id.values[0], target_sessions[0])
        if group.session_id.values[0] == target_sessions[0]:
            print(group.session_id.values[0])
            relevant_idx += list(group.index.values)
            target_sessions.pop(0)

    features_target = features_full.iloc[relevant_idx, :]
    print('Len of resulting features is '.format(len(features_target)))
    path = 'dataset/preprocessed/{}/{}/feature/past_session_features/features.csv'.format(cluster,mode)
    check_folder(path)
    features_target.to_csv(path, index=False)