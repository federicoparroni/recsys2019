from extract_features.feature_base import FeatureBase
import os
import pandas as pd

class ClassifierPiccio(FeatureBase):

    """
    say for each session the type of the last action before clickout if the session is oneshot it is none
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'classifier_piccio'
        super(ClassifierPiccio, self).__init__(
            name=name, mode=mode, cluster=cluster
            )

    def extract_feature(self):
        print('cannot create this features! just move the file feature.csv from drive to the folder \
               \'mode/feature/classifier_piccio/' )

    def read_feature(self, one_hot=False):
        if self.mode == 'full':
            path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
                self.cluster, 'full', self.name)
        else:
            path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
                self.cluster, 'local', self.name)
        if not os.path.exists(path):
            print('cannot find feature, just move the file feature.csv from drive to the folder \
                   \'mode/feature/classifier_piccio/' )


        df = pd.read_csv(path, index_col=None)

        print('{} feature read'.format(self.name))
        return df

