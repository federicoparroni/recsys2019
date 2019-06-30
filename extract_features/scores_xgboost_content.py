from extract_features.feature_base import FeatureBase
import os
import pandas as pd
from sklearn import preprocessing

class ScoresXGBoostContent(FeatureBase):

    """
    say for each session the type of the last action before clickout if the session is oneshot it is none
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'scores_xgboost_content'
        super(ScoresXGBoostContent, self).__init__(
            name=name, mode=mode, cluster=cluster
            )

    def extract_feature(self):
        print('cannot create this features! just move the file feature.csv from drive to the folder \
               \'mode/feature/scores_xgboost_content/' )

    def read_feature(self, one_hot=False, create_not_existing_features=True):
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
                self.cluster, 'local', self.name)
        if not os.path.exists(path):
            print('cannot find feature, just move the file feature.csv from drive to the folder \
                   \'mode/feature/scores_xgboost_content/' )

        df = pd.read_csv(path, index_col=None)
        df = df.drop_duplicates(['user_id','session_id','item_id'], keep='first')

        # Score normalization
        x = df.drop(['user_id', 'session_id', 'item_id'], axis=1)
        x = x.values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df['normalized_xgb_content'] = x_scaled
        df = df[['user_id', 'session_id', 'item_id', 'normalized_xgb_content']]

        print('{} feature read'.format(self.name))
        return df

