from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

class PercClickPerPos(FeatureBase):

    """
    This feature calcuates the popularity on the base of the position in which
    the item was clicked. The idea is that there can be items that appear in low
    positions but are clicked a lot while there are items in first position that
    are not very clicked. The feature computes the popularity for the following
    clusters:
    - popularity in first position (i.e.: # times that an item is in first position
    and it is clicked)
    - pop btw positions 2 to 5
    - pop btw positions 6 to 10
    - pop btw positions 11 to 15
    - pop btw positions 15 to 25
    The popularity is finally divided by the number of times that the item appears
    in those positions.

    The feature format is:

    user_id | session_id | item_id | personalized_popularity

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'perc_click_per_pos'
        super(PercClickPerPos, self).__init__(
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
        # get ALL the clk rows with also last clickouts
        all_clk_rows = df[(df.reference.str.isnumeric()==True)
                  & (df.action_type=='clickout item')][['user_id','session_id','reference','impressions']]
        all_clk_rows.impressions = all_clk_rows.impressions.str.split('|')
        # add the position
        pos_col = get_pos(all_clk_rows.reference.values,all_clk_rows.impressions.values)
        all_clk_rows['position'] = pos_col
        all_clk_rows = all_clk_rows.drop('impressions',1)
        # compute the popularity for each cluster
        df_clicks_1_pos = (
                    all_clk_rows[all_clk_rows.position==1]
                    .groupby("reference")
                    .size()
                    .reset_index(name="pop_1_pos")
                )
        df_clicks_1_pos.reference = df_clicks_1_pos.reference.astype(int)
        df_clicks_1_pos = df_clicks_1_pos.rename(columns={'reference':'item_id'})
        # pos 2 to 5
        df_clicks_2to5_pos = (
            all_clk_rows[(all_clk_rows.position>1)&(all_clk_rows.position<=5)]
            .groupby("reference")
            .size()
            .reset_index(name="pop_2to5_pos")
        )
        df_clicks_2to5_pos.reference = df_clicks_2to5_pos.reference.astype(int)
        df_clicks_2to5_pos = df_clicks_2to5_pos.rename(columns={'reference':'item_id'})
        # pos 6 to 10
        df_clicks_6to10_pos = (
            all_clk_rows[(all_clk_rows.position>5)&(all_clk_rows.position<=10)]
            .groupby("reference")
            .size()
            .reset_index(name="pop_6to10_pos")
        )
        df_clicks_6to10_pos.reference = df_clicks_6to10_pos.reference.astype(int)
        df_clicks_6to10_pos = df_clicks_6to10_pos.rename(columns={'reference':'item_id'})
        # pos 11 to 15
        df_clicks_11to15_pos = (
            all_clk_rows[(all_clk_rows.position>10)&(all_clk_rows.position<=15)]
            .groupby("reference")
            .size()
            .reset_index(name="pop_11to15_pos")
        )
        df_clicks_11to15_pos.reference = df_clicks_11to15_pos.reference.astype(int)
        df_clicks_11to15_pos = df_clicks_11to15_pos.rename(columns={'reference':'item_id'})
        # pos 16 to 25
        df_clicks_16to25_pos = (
            all_clk_rows[(all_clk_rows.position>15)&(all_clk_rows.position<=25)]
            .groupby("reference")
            .size()
            .reset_index(name="pop_16to25_pos")
        )
        df_clicks_16to25_pos.reference = df_clicks_16to25_pos.reference.astype(int)
        df_clicks_16to25_pos = df_clicks_16to25_pos.rename(columns={'reference':'item_id'})
        # now merge with the last clickouts expanded
        last_clickout_indices = find(df)
        last_clk_rows = df.loc[last_clickout_indices, ['user_id','session_id','reference','impressions']]
        last_clk_rows['imp_list'] = last_clk_rows.impressions.str.split('|')
        clk_expanded = expand_impressions(last_clk_rows)
        clk_expanded = clk_expanded.drop('index',1)
        # add position
        pos_col = get_pos(clk_expanded.item_id.values,clk_expanded.imp_list.values)
        clk_expanded['position'] = pos_col
        clk_expanded = clk_expanded.drop('imp_list',1)
        # merge :)
        merged = pd.merge(clk_expanded, df_clicks_1_pos, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_clicks_2to5_pos, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_clicks_6to10_pos, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_clicks_11to15_pos, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_clicks_16to25_pos, how='left',on='item_id').fillna(0)
        # CIAO PICCIO
        # add column of popularity per cluster
        new_col = []
        for t in tqdm(zip(merged.position, merged.pop_1_pos, merged.pop_2to5_pos, merged.pop_6to10_pos,
                          merged.pop_11to15_pos, merged.pop_16to25_pos)):
            if t[0]==1:
                new_col.append(t[1])
            elif 1<t[0]<=5:
                new_col.append(t[2])
            elif 5<t[0]<=10:
                new_col.append(t[3])
            elif 10<t[0]<=15:
                new_col.append(t[4])
            elif 15<t[0]<=25:
                new_col.append(t[5])
        merged['pop_per_pos'] = new_col
        merged = merged.drop(['pop_1_pos','pop_2to5_pos','pop_6to10_pos','pop_11to15_pos','pop_16to25_pos'],axis=1)
        # now compute the number of time that each item is impressed for each cluster position
        all_clks = df[(df.reference.str.isnumeric()==True)
              & (df.action_type=='clickout item')][['user_id','session_id','impressions']]
        all_clks['imp_list'] = all_clks.impressions.str.split('|')
        all_clk_rows_expanded = expand_impressions(all_clks)
        pos_col = get_pos(all_clk_rows_expanded.item_id.values,all_clk_rows_expanded.imp_list.values)
        all_clk_rows_expanded['position'] = pos_col
        # first pos
        all_clk_rows_expanded = all_clk_rows_expanded[['user_id','session_id','item_id','position']]
        df_impressions_1 = (
                    all_clk_rows_expanded[all_clk_rows_expanded.position==1]
                    .groupby(["item_id"])
                    .size()
                    .reset_index(name="n_times_in_position_1")
                )
        df_impressions_1.reference = df_impressions_1.item_id.astype(int)
        # pos 2 to 5
        df_impressions_2to5 = (
            all_clk_rows_expanded[(all_clk_rows_expanded.position>1)&(all_clk_rows_expanded.position<=5)]
            .groupby(["item_id"])
            .size()
            .reset_index(name="n_times_in_position_2to5")
        )
        df_impressions_2to5.reference = df_impressions_2to5.item_id.astype(int)
        # pos 6 to 10
        df_impressions_6to10 = (
            all_clk_rows_expanded[(all_clk_rows_expanded.position>5)&(all_clk_rows_expanded.position<=10)]
            .groupby(["item_id"])
            .size()
            .reset_index(name="n_times_in_position_6to10")
        )
        df_impressions_6to10.reference = df_impressions_6to10.item_id.astype(int)
        # pos 11 to 15
        df_impressions_11to15 = (
            all_clk_rows_expanded[(all_clk_rows_expanded.position>10)&(all_clk_rows_expanded.position<=15)]
            .groupby(["item_id"])
            .size()
            .reset_index(name="n_times_in_position_11to15")
        )
        df_impressions_11to15.reference = df_impressions_11to15.item_id.astype(int)
        # pos 16 to 25
        df_impressions_16to25 = (
            all_clk_rows_expanded[(all_clk_rows_expanded.position>15)&(all_clk_rows_expanded.position<=25)]
            .groupby(["item_id"])
            .size()
            .reset_index(name="n_times_in_position_16to25")
        )
        df_impressions_16to25.reference = df_impressions_16to25.item_id.astype(int)
        # merge with the expanded last clickouts
        merged = pd.merge(merged, df_impressions_1, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_impressions_2to5, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_impressions_6to10, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_impressions_11to15, how='left',on='item_id').fillna(0)
        merged = pd.merge(merged, df_impressions_16to25, how='left',on='item_id').fillna(0)
        # add the new column
        new_col = []
        for t in tqdm(zip(merged.position, merged.n_times_in_position_1, merged.n_times_in_position_2to5, merged.n_times_in_position_6to10,
                          merged.n_times_in_position_11to15, merged.n_times_in_position_16to25)):
            if t[0]==1:
                new_col.append(t[1])
            elif 1<t[0]<=5:
                new_col.append(t[2])
            elif 5<t[0]<=10:
                new_col.append(t[3])
            elif 10<t[0]<=15:
                new_col.append(t[4])
            elif 15<t[0]<=25:
                new_col.append(t[5])
        # <3
        merged['n_times_impr'] = new_col
        merged = merged.drop(['n_times_in_position_1','n_times_in_position_2to5',
                              'n_times_in_position_6to10','n_times_in_position_11to15','n_times_in_position_16to25'],1)
        # now compute the feature, remembering:
        # - subtract 1 to the popularity for the clicked items
        # - subtract 1 to each impression position (bc the number of times is calculated on the whole dataset)
        new_col = []
        merged.reference = merged.reference.astype(int)
        merged.item_id = merged.item_id.astype(int)
        for t in tqdm(zip(merged.reference, merged.item_id, merged.pop_per_pos, merged.n_times_impr)):
            if t[3]>1:
                if t[0]==t[1]:
                    new_col.append(((t[2]-1)*100)/(t[3]-1))
                else:
                    new_col.append(((t[2])*100)/(t[3]-1))
            else:
                new_col.append(0)
        merged['perc_click_per_pos'] = new_col

        return merged[['user_id','session_id','item_id','perc_click_per_pos']]

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = PercClickPerPos(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
