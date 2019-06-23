from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from preprocess_utils.remove_last_part_of_clk_sessions import remove_last_part_of_clk_sessions

class SessionNumFilterSel(FeatureBase):

    """
    This feature says for each session the number of filter selections,
    both flat and percentage

    user_id | session_id | item_id | perc_filter_sel

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_num_filter_sel'
        super(SessionNumFilterSel, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # preprocess needed
        df = df.sort_values(by=['user_id','session_id','timestamp','step']).reset_index(drop=True)
        df = remove_last_part_of_clk_sessions(df)
        # compute number of interactions per session
        df_fil_sel = df[df.action_type=='filter selection'][['user_id','session_id','timestamp','step','action_type']]
        feature = (
            df_fil_sel.groupby(['user_id','session_id'])
            .size()
            .reset_index(name='num_filter_sel')
        )
        # compute session length
        sess_size = (
            df.groupby(['user_id','session_id'])
            .size()
            .reset_index(name='session_length')
        )
        # get clk rows and expand
        clickout_rows = df.loc[find(df), ['user_id','session_id','action_type','impressions']][df.action_type == 'clickout item']
        clk_expanded = expand_impressions(clickout_rows).drop(['index','action_type'],1)
        # merge
        final_feature = pd.merge(clk_expanded, feature, how='left', on=['user_id','session_id']).fillna(0)
        final_feature.num_filter_sel = final_feature.num_filter_sel.astype(int)
        final_feature = pd.merge(final_feature, sess_size, how='left', on=['user_id','session_id']).fillna(0)
        final_feature.session_length = final_feature.session_length.astype(int)
        # compute the percentage
        perc = []
        for t in tqdm(zip(final_feature.num_filter_sel, final_feature.session_length)):
            perc.append((t[0]*100)/t[1])
        final_feature['perc_filter_sel'] = perc

        return final_feature[['user_id','session_id','item_id','perc_filter_sel']]

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = SessionNumFilterSel(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
