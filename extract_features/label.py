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
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        #create dataframe with : user_id, session_id, item_id, label (1 if it's the cliked impression, 0 otherwise)
        df=s[['user_id', 'session_id']]
        df[['item_id', 'label']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        print(df)
        return df

if __name__ == '__main__':
    c = ImpressionLabel(mode='small', cluster='no_cluster')
    c.save_feature()
