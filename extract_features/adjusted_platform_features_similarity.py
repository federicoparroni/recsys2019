from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from extract_features.impression_features import ImpressionFeature

from cython_files.cosine_similarity import cosine_similarity

# TODO: usare tutte le interazioni e dare peso doppio ai clickout

class AdjustedPlatformFeaturesSimilarity(FeatureBase):

    """
    Same as PlatformFeaturesSimilarity but using ALL clickouts.
    The fact is that the platform are 55 and the clickouts are A LOT, so the
    chance that a single clickout will cause overfit is very unlikely... In
    order to reduce the chance of overfitting, the cosine similarity is
    computed with a shrink factor of 5 (10 or more may work better...)

    user_id | session_id | item_id | adj_platform_features_similarity

    """

    def __init__(self, mode, metric='cosine', cluster='no_cluster'):
        name = 'adjusted_platform_features_similarity'
        super(AdjustedPlatformFeaturesSimilarity, self).__init__(
            name=name, mode=mode, cluster=cluster)
        self.metric = metric

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # first step: get all the platforms
        platforms = sorted(df.platform.unique().tolist())
        # create df that for each plat will hold the feature vector
        df_plat_feature = pd.DataFrame(columns=['platform','properties_array'])
        df_plat_feature['platform'] = platforms
        # remove last clickouts and do some preprocessing
        last_indices = find(df)
        df_clickout = df[(df.reference.str.isnumeric()==True)&(df['action_type']=='clickout item')][['reference','platform']]
        df_clickout = df_clickout.rename(columns={'reference':'item_id'})
        df_clickout.item_id = df_clickout.item_id.astype(int)
        # get the item metadata in one hot
        o = ImpressionFeature(mode=self.mode)
        df_accomodations = o.read_feature(True)
        df_accomodations = df_accomodations.drop(['properties1 Star', 'properties2 Star',
            'properties3 Star', 'properties4 Star', 'properties5 Star'],1)
        # merge clickouts dataframe with the metadata
        df_clicks_properties = pd.merge(df_clickout, df_accomodations, how='left', on=['item_id'])
        # extract the one hot econded feature into a 1-dim numpy array
        array = df_accomodations.drop(['item_id'],axis=1).values
        # for each item append the features as numpy array
        df_item_features = pd.DataFrame(columns=['item_id','features_array'])
        df_item_features['item_id'] = df_accomodations['item_id'].values
        df_item_features['features_array'] = list(array)
        # for each column compute the sum of all the clickout-rows' features
        new_col = [] # which will hold the platform feature vector
        for p in tqdm(platforms):
            df_clicks_properties_per_plat = df_clicks_properties[df_clicks_properties.platform == p]
            df_clicks_properties_per_plat = df_clicks_properties_per_plat.drop(['item_id','platform'], axis=1)
            df_sum = df_clicks_properties_per_plat.sum()
            # questo if serve perché ci sono plat che non compaiono nei clickout
            # per quelle metto un vettore di 0
            if df_clicks_properties_per_plat.shape[0] !=0:
                df_sum = df_sum.apply(lambda x: x/df_clicks_properties_per_plat.shape[0])
                plat_feature = df_sum.values
            else:
                plat_feature = np.asarray([0]*df_clicks_properties_per_plat.shape[1])
            new_col.append(plat_feature)
        df_plat_feature['properties_array'] = new_col

        # now take the last clickout rows and expand on the impression list
        clickout_rows = df.loc[last_indices, ['user_id','session_id','platform','action_type','impressions']][df.action_type == 'clickout item']
        clk_expanded = expand_impressions(clickout_rows)
        clk_expanded = clk_expanded.drop(['index','action_type'], axis = 1)
        # for each impression, add the feature vector of the platform and the feature vector of the impression
        clk_expanded_wt_plat_feat = pd.merge(clk_expanded, df_plat_feature, how='left', on=['platform'])
        final_feature = pd.merge(clk_expanded_wt_plat_feat, df_item_features, how='left', on=['item_id'])
        # compute the similarity between the impression's feature vector and the plat feature vector
        new_col =[]
        if self.metric == 'cosine':
            shrink = 5  # TRY ME
            for t in tqdm(zip(final_feature.properties_array, final_feature.features_array)):
                new_col.append(cosine_similarity(t[0].astype(np.double), t[1].astype(np.double), shrink))

        final_feature = final_feature[['user_id','session_id','item_id']]
        final_feature['adj_platform_features_similarity'] = new_col
        return final_feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = AdjustedPlatformFeaturesSimilarity(mode, 'cosine', cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
