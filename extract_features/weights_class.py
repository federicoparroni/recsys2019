from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()
from preprocess_utils.last_clickout_indices import expand_impressions

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

        train_df = data.train_df(mode=self.mode, cluster=self.cluster)
        test_df = data.test_df(mode=self.mode, cluster=self.cluster)
        test_df=test_df.fillna(0)
        df_merged = pd.concat([train_df,test_df])
        idxs_click = find_last_clickout_indices(df_merged)
        df = df_merged.loc[idxs_click][['user_id', 'session_id', 'reference', 'impressions']]

        # weights used to balance the classes
        weights = np.array([0.37738, 0.10207, 0.07179, 0.05545, 0.04711, 0.03822, 0.03215, 0.02825, 0.02574,
                            0.02289, 0.02239, 0.02041, 0.01814, 0.01619, 0.01451, 0.01306, 0.01271, 0.01156,
                            0.01174, 0.01072, 0.01018, 0.00979, 0.00858, 0.00868, 0.01029])
        weights = 1 / weights
        wgt_sum = sum(weights)
        weights_array = (weights / wgt_sum)*100
        # weights_array = np.array([0.5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

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
