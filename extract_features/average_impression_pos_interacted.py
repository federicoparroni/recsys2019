from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import os
#os.chdir("../")
#print(os.getcwd())

class ImpressionPositionInteracted(FeatureBase):

    """
    position of the impressions interacted usually when info about impression is available. (number from 1 to 25)
    also position of the last impression interacted/clicked (this hopes to let apply what lazy user recommender does) (-1 is not available)
    | user_id | session_id | frequent_position_visited | last_pos_interacted
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'average_impression_pos_interacted_items'
        super(ImpressionPositionInteracted, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']

            impressions_pos_available = y[y['impressions'] != None][["impressions"]].drop_duplicates()
            # [13, 43, 4352, 543, 345, 3523] impressions

            # Then create dict impression-position
            # {13: 1, 43: 2, ... }

            tuples_impr_pos = []
            for i in impressions_pos_available.index:
                impr = list(map(int, impressions_pos_available.at[i, 'impressions'].split('|')))
                tuples_impr_pos += [(impr[idx], idx+1) for idx in range(len(impr))]

            dict_impr_pos = dict(list(set(tuples_impr_pos)))

            sum_pos_impr = 0
            count_interacted = 0

            # IMPORTANT: I decided to consider impressions and clickouts distinctively.
            # If an impression is also clicked, that price counts double
            df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][
                ["reference", "impressions", "action_type"]].drop_duplicates()

            # Not considering last clickout in the train sessions
            clks_num_reference = df_only_numeric[(df_only_numeric['action_type'] == 'clickout item') & (pd.to_numeric(df_only_numeric['reference'], errors='coerce').notnull())]

            if len(y) > 0 and len(clks_num_reference) == len(y):  # is it a train or test session? Drop last clickout
                idx_last_clk = y.tail(1).index.values[0]
                df_only_numeric = df_only_numeric.drop(idx_last_clk)

            for i in df_only_numeric.index:
                reference = int(df_only_numeric.at[i, 'reference'])
                if reference in dict_impr_pos.keys():
                    sum_pos_impr += int(dict_impr_pos[reference])
                    count_interacted += 1

            mean_pos = -1
            pos_last_reference = -1
            if count_interacted > 0:
                mean_pos = round(sum_pos_impr / count_interacted, 2)
                last_reference = int(df_only_numeric.tail(1).reference)
                if last_reference in dict_impr_pos.keys():
                    pos_last_reference = int(dict_impr_pos[last_reference])


            return mean_pos, pos_last_reference

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)

        return pd.DataFrame({'user_id': [x[0] for x in s.index.values], 'session_id': [x[1] for x in s.index.values],
                             'mean_pos_interacted': [x[0] for x in s.values], 'pos_last_interaction': [x[1] for x in s.values]})

if __name__ == '__main__':
    c = ImpressionPositionInteracted(mode='local', cluster='cluster_sessions_no_numerical_reference')
    c.save_feature()
