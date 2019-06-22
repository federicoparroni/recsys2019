from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from preprocess_utils.remove_last_part_of_clk_sessions import remove_last_part_of_clk_sessions

class SessionNumNotNumeric(FeatureBase):

    """
    This feature says for each session the flat number and the percentage of non
    numeric interactions in that session

    user_id | session_id | item_id | perc_not_numeric

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_num_not_numeric'
        super(SessionNumNotNumeric, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # preprocess needed
        df = df.sort_values(by=['user_id','session_id','timestamp','step']).reset_index(drop=True)
        df = remove_last_part_of_clk_sessions(df)

        sess_not_numeric_interactions = (
            df[df.reference.str.isnumeric()!=True][['user_id','session_id','timestamp','step']].groupby(['user_id','session_id'])
            .size()
            .reset_index(name='num_not_numeric_interactions')
        )

        sess_size = (
            df.groupby(['user_id','session_id'])
            .size()
            .reset_index(name='session_length')
        )

        clickout_rows = df.loc[find(df), ['user_id','session_id','action_type','impressions']][df.action_type == 'clickout item']
        clk_expanded = expand_impressions(clickout_rows).drop('index',1)

        feature = pd.merge(clk_expanded, sess_not_numeric_interactions, how='left', on=['user_id','session_id']).fillna(0)
        feature.num_not_numeric_interactions = feature.num_not_numeric_interactions.astype(int)
        feature = pd.merge(feature, sess_size, how='left', on=['user_id','session_id']).fillna(0)
        feature.session_length = feature.session_length.astype(int)
        perc = []
        for t in tqdm(zip(feature.num_not_numeric_interactions, feature.session_length)):
            perc.append((t[0]*100)/t[1])
        feature['perc_not_numeric'] = perc

        return feature[['user_id','session_id','item_id','perc_not_numeric']]

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = SessionNumNotNumeric(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
