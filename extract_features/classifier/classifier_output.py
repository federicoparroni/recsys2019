import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

from extract_features.top_pop_interaction_image_per_impression import TopPopInteractionImagePerImpression
from recommenders.XGBoost_Classifier import XGBoostWrapperClassifier

tqdm.pandas()


class ClassifierOutput(FeatureBase):
    """
    | user_id | session_id | probability_of_class_1
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'classifier_output'
        super(ClassifierOutput, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        model = XGBoostWrapperClassifier(mode=mode, cluster='no_cluster')
        model.fit()
        feature = model.extract_feature()
        return feature


if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = ClassifierOutput(mode=mode, cluster='no_cluster')
    c.save_feature()
