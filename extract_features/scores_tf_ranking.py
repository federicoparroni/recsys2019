from extract_features.feature_base import FeatureBase
import os
import pandas as pd

class ScoresXGBoostAccomodation(FeatureBase):

    """
    say for each session the type of the last action before clickout if the session is oneshot it is none
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'scores_tf_ranking'
        super(ScoresXGBoostAccomodation, self).__init__(
            name=name, mode=mode, cluster=cluster
            )

    def extract_feature(self):
        print('cannot create this features! just move the file feature.csv from drive to the folder \
               \'mode/feature/scores_tf_ranking/' )

    def read_feature(self, one_hot=False, create_not_existing_features=True):
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
                self.cluster, 'local', self.name)
        if not os.path.exists(path):
            print('cannot find feature, just move the file feature.csv from drive to the folder \
                   \'mode/feature/scores_tf_ranking/' )

        df = pd.read_csv(path, index_col=None)
        df = df.drop_duplicates(['user_id','session_id','item_id'], keep='first')
        df = df.rename(columns={"score" : "score_tf_ranking"})

        print('{} feature read'.format(self.name))
        return df[['user_id', 'session_id', 'scores_tf_ranking']]

