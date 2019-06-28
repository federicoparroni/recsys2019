from extract_features.feature_base import FeatureBase
import os
import pandas as pd

class ScoresSupersampling(FeatureBase):

    """
    say for each session the type of the last action before clickout if the session is oneshot it is none
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'scores_supersampling'
        super(ScoresSupersampling, self).__init__(
            name=name, mode=mode, cluster=cluster
            )

    def extract_feature(self):
        print('cannot create this features! just move the file feature.csv from drive to the folder \
               \'mode/feature/scores_supersampling/' )

    def read_feature(self, one_hot=False, create_not_existing_features=True):
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
                self.cluster, 'local', self.name)
        if not os.path.exists(path):
            print('cannot find feature, just move the file feature.csv from drive to the folder \
                   \'mode/feature/scores_supersampling/' )

        df = pd.read_csv(path, index_col=None)
        df = df.drop_duplicates(['user_id','session_id','item_id'], keep='first')

        print('{} feature read'.format(self.name))
        return df

