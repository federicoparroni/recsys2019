from extract_features.feature_base import FeatureBase
import data
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
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

class ImpressionLabel(FeatureBase):

    """
    say for each impression of a clickout if it is the one clicked (1) or no 0
    | user_id | session_id | item_id | label
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_label'
        super(ImpressionLabel, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                clicked_impr = clk.reference.values[0]
                for i in impr:
                    if i == clicked_impr:
                        r.append((i,1))
                    else:
                        r.append((i,0))
            return r


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


        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        if self.mode in ['small', 'local']:
            print('reinserting clickout')
            test = test.groupby(['session_id', 'user_id']).progress_apply(_reinsert_clickout)
        df = pd.concat([train, test])
        idxs_click = find_last_clickout_indices(df)
        df = df.loc[idxs_click][['user_id', 'session_id', 'reference', 'impressions']]
        df = expand_impressions(df)
        df['label'] = (df['item_id'] == df['reference'].astype('float'))*1
        df.drop(['index', 'reference'], axis=1, inplace=True)
        print('len of df')
        print(len(df))
        print('len of groups')
        print(len(df.groupby(['user_id', 'session_id', 'item_id'])))

        print(df)
        return df

if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = ImpressionLabel(mode=mode, cluster='no_cluster')
    c.save_feature()
