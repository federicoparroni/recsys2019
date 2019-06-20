from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()


class ActionsInvolvingImpressionSession(FeatureBase):

    """
    the kind of last actions with which the user interacted with the impression before the clickout

    | user_id | session_id | item_id | actions_involving_impression_session_clickout_item |
      actions_involving_impression_session_interaction_item_deals | ... | actions_involving_impression_session_no_action

    for any kind of action with potential numeric reference, the count of the times the user interacted with
    the impression in the context of that action
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'actions_involving_impression_session'
        super(ActionsInvolvingImpressionSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                r = []
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                x = x.loc[head_index.values[0]:clk.index.values[0]-1]
                df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][[
                    "reference", "action_type", "frequence"]]
                refs = list(df_only_numeric.reference.values)
                freqs = list(df_only_numeric.frequence.values)
                actions = list(df_only_numeric.action_type.values)
                count = 0
                for i in impr:
                    actions_involving_impression_session_clickout_item = 0
                    actions_involving_impression_session_interaction_item_deals = 0
                    actions_involving_impression_session_interaction_item_image = 0
                    actions_involving_impression_session_interaction_item_info = 0
                    actions_involving_impression_session_interaction_item_rating = 0
                    actions_involving_impression_session_search_for_item = 0
                    actions_involving_impression_session_no_action = 0
                    if i in refs:
                        occ = [j for j, x in enumerate(refs) if x == i]
                        for o in occ:
                            if actions[o] == 'clickout item':
                                actions_involving_impression_session_clickout_item += freqs[o]
                            elif actions[o] == 'interaction item deals':
                                actions_involving_impression_session_interaction_item_deals += freqs[o]
                            elif actions[o] == 'interaction item image':
                                actions_involving_impression_session_interaction_item_image += freqs[o]
                            elif actions[o] == 'interaction item info':
                                actions_involving_impression_session_interaction_item_info += freqs[o]
                            elif actions[o] == 'interaction item rating':
                                actions_involving_impression_session_interaction_item_rating += freqs[o]
                            elif actions[o] == 'search for item':
                                actions_involving_impression_session_search_for_item += freqs[o]
                    else:
                        actions_involving_impression_session_no_action += 1
                    r.append((i, actions_involving_impression_session_clickout_item,
                             actions_involving_impression_session_interaction_item_deals,
                             actions_involving_impression_session_interaction_item_image,
                             actions_involving_impression_session_interaction_item_info,
                             actions_involving_impression_session_interaction_item_rating,
                             actions_involving_impression_session_search_for_item,
                             actions_involving_impression_session_no_action))
                    count += 1
                return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        df = df.drop(['timestamp', 'step', 'platform', 'city', 'device', 'current_filters', 'prices'], axis=1)
        s = df.groupby(
            ['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars=['user_id', 'session_id'], value_name='tuple').sort_values(
            by=['user_id', 'session_id']).dropna()
        s[['item_id', 'actions_involving_impression_session_clickout_item',
           'actions_involving_impression_session_interaction_item_deals',
           'actions_involving_impression_session_interaction_item_image',
           'actions_involving_impression_session_interaction_item_info',
           'actions_involving_impression_session_interaction_item_rating',
           'actions_involving_impression_session_search_for_item',
           'actions_involving_impression_session_no_action']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        s = s.reset_index(drop=True)
        return s


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection
    cluster = cluster_selection()
    mode = mode_selection()
    c = ActionsInvolvingImpressionSession(mode=mode, cluster=cluster)
    c.save_feature()
