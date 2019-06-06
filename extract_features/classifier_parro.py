from extract_features.feature_base import FeatureBase
import os
import pandas as pd

class ClassifierParro(FeatureBase):

    """
    say for each session the type of the last action before clickout if the session is oneshot it is none
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'classifier_parro'
        super(ClassifierParro, self).__init__(
            name=name, mode=mode, cluster=cluster
            )

    def extract_feature(self):
        print('cannot create this features! just move the file feature.csv from drive to the folder \
               \'mode/feature/classifier_parro/' )

    def read_feature(self, one_hot=False):
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
                self.cluster, 'full', self.name)
        if not os.path.exists(path):
            print('cannot find feature, just move the file feature.csv from drive to the folder \
                   \'mode/feature/classifier_parro/' )

        df = pd.read_csv(path, index_col=0).reset_index(drop=True)
        df = df.groupby(['user_id','session_id'], as_index=False).last()
        print('{} feature read'.format(self.name))
        return df
