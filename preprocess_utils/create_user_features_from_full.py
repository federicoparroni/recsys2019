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
    print('Len of resulting features is:')
    print(len(features_target))

    print('Correcting feature - delete same name sessions')

    feat_label = pd.read_csv('dataset/preprocessed/no_cluster/{}/feature/impression_label/features.csv'.format(mode))
    print(len(feat_label))

    sessions_to_correct = ['2480cd59859f7', '2a181b2125efe', '35e9a348c9d07', '48880df1f1ac9', '54418e6c20dd6',
                           '5ac1377bb44ec',
                           '5ea73da580d41', '711fc00536723', '7e5ec0a512233', 'a14268acc47ab', 'a7255a848e9df',
                           'a940556420c16', 'a9bd5353ae089',
                           'b76440eac54b3', 'd8198666e22d6', 'e6eb492282abf']

    to_add = feat_label[feat_label.session_id.isin(sessions_to_correct)]
    to_add = to_add.drop(['label'], axis=1)
    cols = 'past_times_interacted_impr,past_session_num,past_time_from_closest_interaction_impression,past_times_user_interacted_impression,past_actions_involving_impression_session_clickout_item,past_actions_involving_impression_session_item_image,past_actions_involving_impression_session_item_rating,past_actions_involving_impression_session_item_deals,past_actions_involving_impression_session_item_info,past_actions_involving_impression_session_search_for_item,past_actions_involving_impression_session_no_action,past_mean_price_interacted,past_mean_cheap_pos_interacted,past_mean_pos,past_pos_closest_reference,past_position_impression_changed_closest_clickout,future_times_interacted_impr,future_session_num,future_time_from_closest_interaction_impression,future_times_user_interacted_impression,future_actions_involving_impression_session_clickout_item,future_actions_involving_impression_session_item_image,future_actions_involving_impression_session_item_rating,future_actions_involving_impression_session_item_deals,future_actions_involving_impression_session_item_info,future_actions_involving_impression_session_search_for_item,future_actions_involving_impression_session_no_action,future_mean_price_interacted,future_mean_cheap_pos_interacted,future_mean_pos,future_pos_closest_reference,future_position_impression_changed_closest_clickout'.split(',')
    to_add[cols] = pd.DataFrame([[-1] * 32], index=to_add.index)

    features_target = features_target[~features_target.session_id.isin(sessions_to_correct)]
    features_target = pd.concat([features_target, to_add])
    features_target = features_target.sort_values(by=['user_id', 'session_id'])

    print(len(features_target))

    path = 'dataset/preprocessed/{}/{}/feature/past_session_features/features.csv'.format(cluster, mode)
    check_folder(path)

    features_target.to_csv(path, index=False)
