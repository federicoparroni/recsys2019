from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
tqdm.pandas()


class StatisticsTimeFromLastAction(FeatureBase):

    """
    length of a session:
    user_id | session_id | elapsed_last_action_click_log | variance_last_action | std_last_action
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'statistics_time_from_last_action'
        super(StatisticsTimeFromLastAction, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        idxs_click = find_last_clickout_indices(df)
        temp = df[['user_id', 'session_id', 'step', 'timestamp']]
        session_id_l = []
        length_step_l = []
        length_timestamp_l = []
        timestamp_last_action_l = []
        final_timestamp_l = []
        user_id_l = []
        for i in tqdm(idxs_click):
            user_id = temp.at[i, 'user_id']
            session_id = temp.at[i, 'session_id']
            step = temp.at[i, 'step']
            f_timestamp = temp.at[i, 'timestamp']
            i_timestamp = temp.at[i - (step - 1), 'timestamp']
            if step > 1:
                timestamp_last_action = temp.at[i - 1, 'timestamp']
            else:
                timestamp_last_action = f_timestamp

            user_id_l.append(user_id)
            session_id_l.append(session_id)
            length_step_l.append(int(step))
            length_timestamp_l.append(int(f_timestamp - i_timestamp))
            timestamp_last_action_l.append(int(timestamp_last_action))
            final_timestamp_l.append(int(f_timestamp))
        final_df = pd.DataFrame({'user_id': user_id_l, 'session_id': session_id_l, 'length_step': length_step_l,
                                 'length_timestamp': length_timestamp_l,
                                 'timestamp_last_action': timestamp_last_action_l,
                                 'final_timestamp': final_timestamp_l})
        final_df['mean_time_action'] = final_df['length_timestamp'] / final_df['length_step']

        final_df['elapsed_last_action_click'] = final_df['final_timestamp'] - final_df['timestamp_last_action']

        final_df['elapsed_last_action_click_log'] = np.log(final_df['elapsed_last_action_click'] + 1)

        final_df['variance_last_action'] = (final_df['elapsed_last_action_click'] - final_df['mean_time_action']) ** 2

        final_df['std_last_action'] = abs(final_df['elapsed_last_action_click'] - final_df['mean_time_action'])

        final_df.drop(['timestamp_last_action', 'final_timestamp', 'mean_time_action', \
                       'length_step', 'length_timestamp', 'elapsed_last_action_click'], axis=1, inplace=True)
        return final_df


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = StatisticsTimeFromLastAction(mode=mode, cluster=cluster)
    c.save_feature()
