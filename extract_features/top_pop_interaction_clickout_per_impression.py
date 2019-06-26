from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()
from extract_features.impression_features import ImpressionFeature
from preprocess_utils.last_clickout_indices import find


class TopPopInteractionClickoutPerImpression(FeatureBase):

    """
    say for each impression of a clickout the popularity of the impression ie the number of times a user
    made a clickout on one it
    | item_id | top_pop_interaction_clickout_per_impression
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'top_pop_interaction_clickout_per_impression'
        super(TopPopInteractionClickoutPerImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        o = ImpressionFeature(self.mode)
        f = o.read_feature()
        f = f.drop(['properties'], axis=1)
        f['popularity'] = 0
        pop = dict(zip(f.item_id.values, f.popularity.values))

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        last_clickout_indices = find(df)
        df_dropped_last_clickouts = df.drop(last_clickout_indices)
        df_no_last_clickouts = df_dropped_last_clickouts[(df_dropped_last_clickouts.action_type == 'clickout item') & ~(df_dropped_last_clickouts.reference.isnull())]
        references = df_no_last_clickouts.reference.values

        for r in references:
            pop[int(r)] += 1

        final_df = pd.DataFrame(list(pop.items()), columns=['item_id', 'top_pop_interaction_clickout_per_impression'])

        return final_df

    def post_loading(self, df):
        df.top_pop_interaction_clickout_per_impression = df.top_pop_interaction_clickout_per_impression.astype(np.float)
        v = df.top_pop_interaction_clickout_per_impression.values
        v += 1
        df.top_pop_interaction_clickout_per_impression = np.log(v)
        return df

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = TopPopInteractionClickoutPerImpression(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    # print(c.read_feature(one_hot=True))
