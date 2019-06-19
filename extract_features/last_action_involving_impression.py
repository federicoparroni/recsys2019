from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from preprocess_utils.remove_last_part_of_clk_sessions import remove_last_part_of_clk_sessions

class LastActionInvolvingImpression(FeatureBase):

    """
    FAST VERSION (se non funziona prendetevela con @teomore)
    Specifies the kind of last actions with which the user interacted with the
    impression before the clickout

    | user_id | session_id | item_id | last_action_involving_impression

    last_action_involving_impression is a string
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'last_action_involving_impression'
        columns_to_onehot = [('last_action_involving_impression', 'single')]
        super(LastActionInvolvingImpression, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # remove last clks and last part of session
        new_df = remove_last_part_of_clk_sessions(df)
        new_df = new_df.drop(find(new_df))
        no_last_clks_numeric = new_df[new_df.reference.str.isnumeric()==True][['user_id','session_id','action_type','reference']]
        # we want to make it fast, avoid any loops...
        # simply drop duplicates and mantain last occurence
        # of the tuple user-session-item :D
        last_actions = no_last_clks_numeric.drop_duplicates(['user_id','session_id','reference'], keep='last')
        last_actions = last_actions.rename(columns={'reference':'item_id',
                                            'action_type':'last_action_involving_impression'})
        last_actions.item_id = last_actions.item_id.astype(int)
        # get last clickouts and expand
        last_clk = df.loc[find(df)]
        clk_expanded = expand_impressions(last_clk)[['user_id','session_id','item_id']]
        # now simply merge and fill NaNs with 'no_action' as in the original feature
        feature = pd.merge(clk_expanded, last_actions, how='left',on=['user_id','session_id','item_id'])
        feature.last_action_involving_impression = feature.last_action_involving_impression.astype(object).fillna('no_action')
        return feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = LastActionInvolvingImpression(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
