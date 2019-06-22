from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class SessionActionNumRefDiffFromImpressions(FeatureBase):

    """
    | user_id | session_id | actions_num_ref_diff_from_impressions_clickout_item | actions_num_ref_diff_from_impressions_interaction_item_deals | ..

    tells how many time the user clicked during a session in an action with numeric ref diff from all the other
    impressions
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_action_num_ref_diff_from_impressions'
        super(SessionActionNumRefDiffFromImpressions, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            c = 0
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clickout_item = 0
                interaction_item_deals = 0
                interaction_item_image = 0
                interaction_item_info = 0
                interaction_item_rating = 0
                search_for_item = 0
                no_action = 0
                clk = y.tail(1)
                impr = clk.impressions.values[0]
                x = x[x['step'] < int(clk['step'])]
                df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()]
                refs = list(df_only_numeric.reference.values)
                freqs = list(df_only_numeric.frequence.values)
                actions = list(df_only_numeric.action_type.values)
                for i in range(len(refs)):
                    if refs[i] not in impr:
                        action_i = actions[i]
                        if action_i == 'clickout item':
                            clickout_item += int(freqs[i])
                        elif action_i == 'interaction item deals':
                            interaction_item_deals += int(freqs[i])
                        elif action_i == 'interaction item info':
                            interaction_item_info += int(freqs[i])
                        elif action_i == 'interaction item image':
                            interaction_item_image += int(freqs[i])
                        elif action_i == 'interaction item rating':
                            interaction_item_rating += int(freqs[i])
                        elif action_i == 'search for item':
                            search_for_item += int(freqs[i])
                        c += 1
                if c == 0:
                    no_action += 1
                return [(clickout_item, interaction_item_deals, interaction_item_image,
                        interaction_item_info, interaction_item_rating, search_for_item, no_action)]

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        df = df[["session_id", "user_id", "reference", "step", "action_type", "frequence", "impressions"]]
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars=['user_id', 'session_id'], value_name='tuple').sort_values(
            by=['user_id', 'session_id']).dropna()
        s[['actions_num_ref_diff_from_impressions_clickout_item',
           'actions_num_ref_diff_from_impressions_interaction_item_deals',
           'actions_num_ref_diff_from_impressions_interaction_item_image',
           'actions_num_ref_diff_from_impressions_interaction_item_info',
           'actions_num_ref_diff_from_impressions_interaction_item_rating',
           'actions_num_ref_diff_from_impressions_search_for_item',
           'actions_num_ref_diff_from_impressions_no_action']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        return s


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = SessionActionNumRefDiffFromImpressions(
        mode=mode, cluster=cluster)
    c.save_feature()
