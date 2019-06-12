from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions


class ImpressionPositionInPercentage(FeatureBase):

    """
    This features tells, for each impression in the clickout, the position in
    percentage wrt to the total number of impressions of that clickout

    user_id | session_id | item_id | impression_position_in_percentage

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_position_in_percentage'
        super(ImpressionPositionInPercentage, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # get clickout rows
        clickout_rows = df.loc[find(df), ['user_id','session_id','impressions']][df.action_type == 'clickout item']
        clk_expanded = expand_impressions(clickout_rows).drop(['index'],1)
        # get position
        new_col = []
        curr_u = clk_expanded.loc[0,'user_id']
        curr_s = clk_expanded.loc[0,'session_id']
        pos = 0
        for t in tqdm(zip(clk_expanded.user_id, clk_expanded.session_id)):
            if t[0]==curr_u and t[1]==curr_s:
                pos +=1
            else:
                pos = 1
                curr_u = t[0]
                curr_s = t[1]
            new_col.append(pos)
        clk_expanded['position'] = new_col
        # get impression count for each session
        imp_count = (
            clk_expanded.groupby(['user_id','session_id'])
            .size()
            .reset_index(name='num_impressions')
        )
        # merge and compute percentage
        feature = pd.merge(clk_expanded, imp_count, how='left',on=['user_id','session_id']).fillna(0)
        pos_perc = []
        for t in tqdm(zip(feature.position, feature.num_impressions)):
            pos_perc.append((t[0]*100)/t[1])
        feature['impression_position_in_percentage'] = pos_perc

        return feature[['user_id','session_id','item_id','impression_position_in_percentage']]

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = ImpressionPositionInPercentage(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
