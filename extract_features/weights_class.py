from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()

def _reinsert_clickout(df):
    # take the row of the missing clickout
    clickout_rows_df = df[(df['action_type'] == 'clickout item') & df['reference'].isnull()]
    # check if it exsists
    if len(clickout_rows_df)>0:
        # retrieve from the full_df the clickout
        missing_click = data.full_df().loc[clickout_rows_df.index[0]]['reference']
        # reinsert the clickout on the df
        df.at[clickout_rows_df.index[0], 'reference']= missing_click
    return df

class WeightsClass(FeatureBase):

    """
    weights to balance the classes
    | user_id | session_id | item_id | weights
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'weights_class'
        super(WeightsClass, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def find_last_clickout_indices(df):
            indices = []
            cur_ses = ''
            cur_user = ''
            temp_df = df[df.action_type == 'clickout item'][['user_id', 'session_id', 'action_type']]
            for idx in tqdm(temp_df.index.values[::-1]):
                ruid = temp_df.at[idx, 'user_id']
                rsid = temp_df.at[idx, 'session_id']
                if (ruid != cur_user or rsid != cur_ses):
                    indices.append(idx)
                    cur_user = ruid
                    cur_ses = rsid
            return indices[::-1]

        def expand_impressions(df):
            res_df = df.copy()
            res_df.impressions = res_df.impressions.str.split('|')
            res_df = res_df.reset_index()

            res_df = pd.DataFrame({
                col: np.repeat(res_df[col].values, res_df.impressions.str.len())
                for col in res_df.columns.drop('impressions')}
            ).assign(**{'impressions': np.concatenate(res_df.impressions.values)})[res_df.columns]

            res_df = res_df.rename(columns={'impressions': 'item_id'})
            res_df = res_df.astype({'item_id': 'int'})

            return res_df


        train_df = data.train_df(mode=self.mode, cluster=self.cluster)
        idxs_click = find_last_clickout_indices(train_df)
        df = train_df.loc[idxs_click][['user_id', 'session_id', 'reference', 'impressions']]

        # weights used to balance the classes
        weights = np.array([0.37738, 0.10207, 0.07179, 0.05545, 0.04711, 0.03822, 0.03215, 0.02825, 0.02574,
                            0.02289, 0.02239, 0.02041, 0.01814, 0.01619, 0.01451, 0.01306, 0.01271, 0.01156,
                            0.01174, 0.01072, 0.01018, 0.00979, 0.00858, 0.00868, 0.01029])
        weights = 1 / weights
        wgt_sum = sum(weights)
        weights_array = (weights / wgt_sum)*100

        weights_column_list = []
        for idx, row in tqdm(df.iterrows()):
            impressions = list(map(int, row.impressions.split('|')))
            reference = int(row.reference)
            if reference in impressions:
                position = impressions.index(reference)
                weights_column_list.append(weights_array[position])
            else:
                # the reference is not in the impressions so weight the sample 0
                weights_column_list.append(0)

        df['weights']=weights_column_list

        df = expand_impressions(df)
        df.drop(['index', 'reference'], axis=1, inplace=True)
        return df

if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = WeightsClass(mode=mode, cluster='no_cluster')
    c.save_feature()
