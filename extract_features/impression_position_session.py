from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
tqdm.pandas()


class ImpressionPositionSession(FeatureBase):

    """
    the position in which an impression shows up in a clickout
    | user_id | session_id | item_id | impression_position
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_position_session'
        super(ImpressionPositionSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        idxs_click = find_last_clickout_indices(df)
        df = df.loc[idxs_click][['user_id', 'session_id', 'impressions']]
        df = expand_impressions(df)
        # initialize the session id
        session_id = ''
        count = 1
        impression_position = []
        for i in tqdm(df.index):
            c_session = df.at[i, 'session_id']
            if c_session != session_id:
                session_id = c_session
                count = 1
            impression_position.append(count)
            count += 1
        df['impression_position'] = impression_position
        df['impression_position']=pd.to_numeric(df['impression_position'])
        df.drop('index', axis=1, inplace=True)

        return df

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection
    mode = mode_selection()
    cluster = cluster_selection()
    c = ImpressionPositionSession(mode=mode, cluster=cluster)
    c.save_feature()
