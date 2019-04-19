from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class ImpressionLabel(FeatureBase):

    """
    say for each impression of a clickout if it is the one clicked (1) or no 0
    | user_id | session_id | item_id | label
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression label'
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
                        r.append(1)
                    else:
                        r.append(0)
                print(r)
            return r
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        #create dataframe with : user_id, session_id, label, item_id
        df = s[['user_id', 'session_id', 'tuple']]
        df['item_id'] = s['tuple'].index[:]
        df.rename(columns={'tuple':'label'}, inplace=True) # renaming the column
        df = df.reset_index(drop=True)
        print(df)
        return df

if __name__ == '__main__':
    c = ImpressionLabel(mode='small', cluster='no_cluster')
    c.save_feature()
