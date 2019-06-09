from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from extract_features.impression_features import ImpressionFeature

from cython_files.cosine_similarity import cosine_similarity

# TODO: usare tutti i clickout tranne quello della sessione?
# TODO: usare tutte le interazioni e dare peso doppio ai clickout

class NormalizedPlatformFeaturesSimilarity(FeatureBase):

    """
    This features tries to model the average user of a platform wrt the global
    average of properties that the user searches.
    It does the following:
    1) takes only the clickouts rows (NOT last click to avoid bias)
    2) for each platform, sum all the features of the clk in that platform
        ---> this will give a feature vector for each plat
    3) compute a global feature vector summing all the clk items features
        ---> this will give a global feature vector
    4) divide each platform vector by the global vector
    5) finally, for each impression of a clickout in that platform computes
    the similarity between the feature vector of the impression and the globally
    normalized feature vector of the platform

    user_id | session_id | item_id | platform_similarity_normalized

    """

    def __init__(self, mode, metric='cosine', cluster='no_cluster'):
        name = 'normalized_platform_features_similarity'
        super(NormalizedPlatformFeaturesSimilarity, self).__init__(
            name=name, mode=mode, cluster=cluster)
        self.metric = metric

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        platforms = df['platform'].unique().tolist()
        df_plat_feature = pd.DataFrame(columns=['platform','properties_array'])
        df_plat_feature['platform'] = platforms
        last_indices = find(df)
        df_non_last_clk = df.drop(last_indices)
        df_clickout = df_non_last_clk[(df_non_last_clk['action_type']=='clickout item')][['reference','platform']]
        df_clickout = df_clickout.rename(columns={'reference':'item_id'})
        df_clickout = df_clickout.dropna() # remove NaNs
        df_clickout.item_id = df_clickout.item_id.astype(int)
        o = ImpressionFeature(mode=self.mode)
        df_accomodations = o.read_feature(True)
        df_accomodations = df_accomodations.drop(['properties1 Star', 'properties2 Star', 'properties3 Star', 'properties4 Star', 'properties5 Star'],1)

        df_clicks_properties = pd.merge(df_clickout, df_accomodations, how='left', on=['item_id'])
        array = df_accomodations.drop(['item_id'],axis=1).values
        df_item_features = pd.DataFrame(columns=['item_id','features_array'])
        df_item_features['item_id'] = df_accomodations['item_id'].values
        df_item_features['features_array'] = list(array)

        new_col = []
        for p in tqdm(platforms):
            df_clicks_properties_per_plat = df_clicks_properties[df_clicks_properties.platform == p]
            df_clicks_properties_per_plat = df_clicks_properties_per_plat.drop(['item_id','platform'], axis=1)
            df_sum = df_clicks_properties_per_plat.sum()
            if df_clicks_properties_per_plat.shape[0] !=0: # questo vuol dire che appare almeno una volta la plat
                plat_feature = df_sum.values
            else:
                plat_feature = np.asarray([0]*df_clicks_properties_per_plat.shape[1])
            new_col.append(plat_feature)

        df_plat_feature['properties_array'] = new_col
        global_sum = df_clicks_properties.drop(['item_id','platform'],1)
        global_sum = global_sum.sum().tolist()

        df_plat_feature['global_properties'] = df_plat_feature.apply(lambda x: global_sum, axis=1)
        properties_globally_normalized = []
        for t in tqdm(zip(df_plat_feature.properties_array, df_plat_feature.global_properties)):
            properties_globally_normalized.append(np.asarray([x/y for x,y in zip(t[0],t[1])]))

        df_plat_feature['properties_globally_normalized'] = properties_globally_normalized
        df_plat_feature = df_plat_feature.drop(['properties_array','global_properties'],1)

        # ora prendo il dataframe coi clickout solito
        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','platform','action_type','impressions']][df.action_type == 'clickout item']
        clk_expanded = expand_impressions(clickout_rows)

        clk_expanded = clk_expanded.drop(['index','action_type'], axis = 1)
        clk_expanded_wt_plat_feat = pd.merge(clk_expanded, df_plat_feature, how='left', on=['platform']).astype(object)
        clk_expanded_wt_plat_feat.item_id = clk_expanded_wt_plat_feat.item_id.astype(int)

        final_feature = pd.merge(clk_expanded_wt_plat_feat, df_item_features, how='left', on=['item_id'])
        new_col =[]
        shrink = 0 # TRY ME
        for t in tqdm(zip(final_feature.properties_globally_normalized, final_feature.features_array)):
            new_col.append(cosine_similarity(t[0].astype(np.double), t[1].astype(np.double),shrink))

        new_feature = final_feature[['user_id','session_id','item_id']]
        new_feature['platform_similarity_normalized'] = new_col

        return new_feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = NormalizedPlatformFeaturesSimilarity(mode, 'cosine', cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
