from extract_features.feature_base import FeatureBase
import data
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()

class ActionTypeBefClick(FeatureBase):

    """
    say for each session the type of the last action before clickout if the session is oneshot it is none
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'action_type_bef_click'
        columns_to_onehot = [('last_action_type_before_click', 'single')]
        super(ActionTypeBefClick, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):

        def _reinsert_clickout(df):
            # take the row of the missing clickout
            clickout_rows_df = df[(df['action_type'] == 'clickout item') & df['reference'].isnull()]
            # check if it exsists
            if len(clickout_rows_df) > 0:
                # retrieve from the full_df the clickout
                missing_click = data.full_df().loc[clickout_rows_df.index[0]]['reference']
                # reinsert the clickout on the df
                df.at[clickout_rows_df.index[0], 'reference'] = missing_click
            return df

        def action_bef_click(df):
            df = df.reset_index()
            actions_bef_click = []
            cur_ses = ''
            cur_user = ''
            temp_df = df[df.action_type == 'clickout item'][['user_id', 'session_id', 'action_type']]
            temp_df = temp_df.sort_index()
            for idx in tqdm(temp_df.index.values[::-1]):
                ruid = temp_df.at[idx, 'user_id']
                rsid = temp_df.at[idx, 'session_id']
                if (ruid != cur_user or rsid != cur_ses):

                    # update the current_user and current_session
                    cur_user = ruid
                    cur_ses = rsid

                    # retrieve the session and user of the row before the click
                    prev_user = df.at[idx - 1, 'user_id']
                    prev_sess = df.at[idx - 1, 'session_id']

                    # initialize the value of act_bef_click is usefull in case the session has only 1 action
                    act_bef_click = 'None'

                    # if the user and session are equal to ruid and rsid the session has len > 1
                    if prev_user == ruid and prev_sess == rsid:
                        act_bef_click = df.at[idx - 1, 'action_type']

                    # append on a list the tuple [('user_id','session_id','action_type')]
                    actions_bef_click.append((prev_user, prev_sess, act_bef_click))

            # create a dataframe from the list of tuples to return
            return pd.DataFrame(actions_bef_click, columns=['user_id', 'session_id', 'last_action_type_before_click'])

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        if self.mode in ['small', 'local']:
            print('reinserting clickout')
            test = test.groupby(['session_id', 'user_id']).progress_apply(_reinsert_clickout)
        df = pd.concat([train, test])
        return action_bef_click(df)

if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = ActionTypeBefClick(mode=mode, cluster='no_cluster')
    c.save_feature()
