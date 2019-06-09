from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from extract_features.impression_features import ImpressionFeature

# NB:
# MIO CYTHON <3 ---> 81k it/s
# SCIPY COSINE ---> 17k it/s   :P
from cython_files.cosine_similarity import cosine_similarity

# TODO: usare tutti i clickout tranne quello della sessione?
# TODO: usare tutte le interazioni e dare peso doppio ai clickout

class LocationFeaturesSimilarity(FeatureBase):

    """
    This features tries to model the average things that the user searchs in a city.
    It does the following for each plat:
    1) takes only the clickouts rows (NOT last click to avoid bias) for that city
    2) sum all the features of those rows and normalize the sum vector
    3) for each impression of a clickout for that city computes the similarity
    (at this time Euclidean Distance or Cosine Similarity) between the feature vector
    of the impression and the feature vector of the city

    user_id | session_id | item_id | city_features_similarity

    """

    def __init__(self, mode, metric='cosine', cluster='no_cluster'):
        name = 'location_features_similarity'
        super(LocationFeaturesSimilarity, self).__init__(
            name=name, mode=mode, cluster=cluster)
        self.metric = metric

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # get all the cities
        cities = df['city'].unique().tolist()
        # get clickout rows (WITHOUT last clk)
        last_indices = find(df)
        df_non_last_clk = df.drop(last_indices)
        df_clickout = df_non_last_clk[(df_non_last_clk['action_type']=='clickout item')][['reference','city']]
        df_clickout = df_clickout.rename(columns={'reference':'item_id'})
        df_clickout = df_clickout.dropna() # remove NaNs, that should not be there anywayss
        df_clickout.item_id = df_clickout.item_id.astype(int)
        # open impressions df
        o = ImpressionFeature(mode='small')
        df_accomodations = o.read_feature(True)
        df_accomodations = df_accomodations.drop(['properties1 Star', 'properties2 Star', 'properties3 Star', 'properties4 Star', 'properties5 Star'],1)
        # get all clicks properties
        df_clicks_properties = pd.merge(df_clickout, df_accomodations, how='left', on=['item_id'])
        df_clicks_properties = df_clicks_properties.sort_values(by=['city'])
        df_clicks_properties = df_clicks_properties.drop('item_id',1)
        # sum all properties per city
        grouped_by_city = df_clicks_properties.groupby('city').sum()
        # create df with city:array_of_features
        df_city_features = pd.DataFrame(columns=['city','properties_array'])
        df_city_features.city = grouped_by_city.index
        df_city_features.properties_array = grouped_by_city.values.tolist()
        # now take last clk df
        clickout_rows = df.loc[last_indices,
                       ['user_id','session_id','city','action_type','impressions']][df.action_type == 'clickout item']
        clk_expanded = expand_impressions(clickout_rows)
        clk_expanded_wt_city_feat = pd.merge(clk_expanded, df_city_features, how='left', on=['city'])
        # create df with item:array_of_features
        array = df_accomodations.drop(['item_id'],axis=1).values
        df_item_features = pd.DataFrame(columns=['item_id','features_array'])
        df_item_features['item_id'] = df_accomodations['item_id'].values
        df_item_features['features_array'] = list(array)
        final_feature = pd.merge(clk_expanded_wt_city_feat, df_item_features, how='left', on=['item_id'])
        for n in tqdm(final_feature[final_feature['properties_array'].isnull()].index.tolist()):
            final_feature.at[n,'properties_array'] = [0]*152
        # cast list to numpy array to use the cosine (it's written for doubles)
        final_feature.properties_array = final_feature.properties_array.progress_apply(lambda x: np.asarray(x))
        # create new column
        new_col =[]
        if self.metric == 'cosine':
            shrink = 0 # TRY ME
            for t in tqdm(zip(final_feature.properties_array, final_feature.features_array)):
                new_col.append(cosine_similarity(t[0].astype(np.double), t[1].astype(np.double),shrink))
        if self.metric == 'euclidean':
            for t in tqdm(zip(final_feature.properties_array, final_feature.features_array)):
                new_col.append(np.linalg.norm(t[0]-t[1]))
        # final feature
        new_feature = final_feature[['user_id','session_id','item_id']]
        new_feature['city_similarity'] = new_col

        return new_feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = LocationFeaturesSimilarity(mode, 'cosine', cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
