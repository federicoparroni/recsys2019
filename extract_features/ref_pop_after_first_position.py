from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

class RefPopAfterFirstPosition(FeatureBase):

    """
    This feature calcuates the popularity (the one "adjusted" as in PersonalizedTopPop)
    only on clickouts that happen on items that are not in the first position

    user_id | session_id | item_id | n_clicks_after_first_pos
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'ref_pop_after_first_position'
        super(RefPopAfterFirstPosition, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def get_pos(item, rec):
            res = np.empty(item.shape)
            for i in tqdm(range(len(item))):
                if str(item[i]) in rec[i]:
                    res[i] = rec[i].index(str(item[i])) + 1
                else:
                    res[i] = -1
            return res.astype(int)

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        last_clickout_indices = find(df)
        all_clk_rows = df[df.reference.str.isnumeric()==True][df.action_type == 'clickout item']
        all_clk_rows = all_clk_rows [['user_id','session_id','reference','impressions']]

        all_clk_rows.impressions = all_clk_rows.impressions.str.split('|')
        pos_col = get_pos(all_clk_rows.reference.values,all_clk_rows.impressions.values)
        all_clk_rows = all_clk_rows.drop('impressions',1)
        all_clk_rows['position'] = pos_col
        all_clk_rows_after_1 = all_clk_rows[all_clk_rows.position>1]

        df_clicks_after_1 = (
            all_clk_rows_after_1
            .groupby(["reference"])
            .size()
            .reset_index(name="n_clicks_per_item")
        )
        df_clicks_after_1.reference = df_clicks_after_1.reference.astype(int)
        df_clicks_after_1 = df_clicks_after_1.rename(columns={'reference':'item_id'})

        last_clk_rows = df.loc[last_clickout_indices, ['user_id','session_id','reference','impressions']]
        last_clk_rows['imp_list'] = last_clk_rows.impressions.str.split('|')
        clk_expanded = expand_impressions(last_clk_rows)
        clk_expanded = clk_expanded.drop('index',1)

        pos_col = get_pos(clk_expanded.item_id.values,clk_expanded.imp_list.values)
        clk_expanded['position'] = pos_col
        clk_expanded = clk_expanded.drop('imp_list',1)

        merged = pd.merge(clk_expanded, df_clicks_after_1, how='left',on='item_id').fillna(0)
        new_col = []
        merged.item_id = merged.item_id.astype(int)
        merged.reference = merged.reference.astype(int)
        for t in tqdm(zip(merged.item_id, merged.reference, merged.position, merged.n_clicks_per_item)):
            if t[0]==t[1] and t[2]>1:
                new_col.append(int(t[3]-1))
            else:
                new_col.append(int(t[3]))

        merged['n_clicks_after_first_pos'] = new_col
        feature = merged[['user_id','session_id','item_id','n_clicks_after_first_pos']]
        return feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = RefPopAfterFirstPosition(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
