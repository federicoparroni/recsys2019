from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from extract_features.impression_features import ImpressionFeature


class PriceQuality(FeatureBase):

    """
    This feature tries to calculate the quality/price tradeoff as:

        1.5*rating + stars
        ___________________
              price

    user_id | session_id | item_id | price_quality
    """

    def __init__(self, mode, metric='cosine', cluster='no_cluster'):
        name = 'price_quality'
        super(PriceQuality, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        # get clk rows
        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','impressions','prices']]
        clk_expanded = expand_impressions(clickout_rows).drop('index',1)

        # open item metadata in one hot
        o = ImpressionFeature(mode=self.mode)
        df_accomodations = o.read_feature(True)

        # get the stars
        feature_stars = df_accomodations[['item_id','properties1 Star', 'properties2 Star', 'properties3 Star', 'properties4 Star', 'properties5 Star']]
        # remap the name
        feature_stars = feature_stars.rename(columns={'properties1 Star':'1', 'properties2 Star':'2', 'properties3 Star':'3', 'properties4 Star':'4','properties5 Star':'5'})
        # set default 0 Stars for those for which the feature is missing
        feature_stars['0'] = pd.Series(np.ones(len(feature_stars), dtype=np.uint8),index=feature_stars.index)
        feature_stars['stars'] = feature_stars[['5','4','3','2','1','0']].idxmax(axis=1)
        feature_stars_restricted = feature_stars[['item_id', 'stars']]
        f_stars = pd.merge(clk_expanded, feature_stars_restricted, how='left', on=['item_id'])
        f_stars['stars'] = f_stars['stars'].astype(int)


        # get the ratings
        f_ratings = df_accomodations[['item_id', 'propertiesExcellent Rating',
                       'propertiesVery Good Rating',
                       'propertiesGood Rating',
                       'propertiesSatisfactory Rating',
                       ]]
        f_ratings['propertiesNo Rating'] = pd.Series(np.ones(len(f_ratings), dtype=np.uint8),
                                                              index=f_ratings.index)
        df = f_ratings.iloc[:, 1:]
        df['fake'] = pd.Series(np.zeros(len(df), dtype=np.uint8), index=df.index)
        cols = df.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        df = df.reindex(columns=cols)
        dff = df.diff(axis=1).drop(['fake'], axis=1)
        dff = dff.astype(int)
        dff.columns = [5, 4, 3, 2, 1]
        f_ratings = f_ratings.drop(f_ratings.columns[1:], axis=1)
        f_ratings['rating'] = dff.idxmax(axis=1)
        f_ratings = pd.merge(f_ratings, feature_stars_restricted, how='left', on=['item_id'])
        df_clk_rat_star = pd.merge(clk_expanded, f_ratings, how='left', on='item_id')


        # expand prices
        df_clk_rat_star.prices = df_clk_rat_star.prices.str.split('|')
        curr_user = '_'
        curr_sess = '_'
        pos = 0
        price_expanded = []
        for t in tqdm(zip(df_clk_rat_star.user_id,df_clk_rat_star.session_id, df_clk_rat_star.prices)):
            #check if in session
            if curr_user != t[0] or curr_sess != t[1]:
                pos = 0
                curr_user = t[0]
                curr_sess = t[1]
            else:
                pos += 1
            price_expanded.append(t[2][pos])
        df_clk_rat_star['price'] = price_expanded
        df_clk_rat_star = df_clk_rat_star.drop(['prices'], 1)
        df_clk_rat_star.stars = df_clk_rat_star.stars.astype(int)


        # fills missing stars values with the mean
        avg = df_clk_rat_star[['user_id','session_id','stars']]
        avg = avg.loc[avg.stars != 0] # va calcolata la media solo sui non zero
        avg = pd.DataFrame(avg.groupby(['user_id','session_id'])['stars'].progress_apply(lambda x: int(x.sum()/x.size))).fillna(0)
        avg = avg.rename(columns={'stars':'stars_avg'})
        avg.stars = avg.stars_avg.astype(int)
        no_stars = df_clk_rat_star.loc[df_clk_rat_star.stars == 0, ['user_id','session_id','item_id']]
        stars_filled = pd.merge(no_stars, avg, how='left', on=['user_id','session_id']).fillna(0)
        stars_filled.stars = stars_filled.stars_avg.astype(int)
        df_clk_rat_star_filled = pd.merge(df_clk_rat_star, stars_filled, how='left', on = ['user_id','session_id','item_id'])
        for t in zip (df_clk_rat_star_filled.stars, df_clk_rat_star_filled.stars_avg, df_clk_rat_star_filled.index):
            if t[0] == 0:
                df_clk_rat_star_filled.at[t[2], 'stars'] = t[1]
        df_clk_rat_star_filled = df_clk_rat_star_filled.drop('stars_avg',1)

        # now fill missing values for rating
        avg = df_clk_rat_star_filled[['user_id','session_id','rating']]
        avg.rating = avg.rating.astype(int)
        avg = avg.loc[avg.rating != 1] # va calcolata la media solo sui non zero
        avg = pd.DataFrame(avg.groupby(['user_id','session_id'])['rating'].progress_apply(lambda x:
            int(x.sum()/x.size))).fillna(0)
        avg = avg.rename(columns={'rating':'rating_avg'})
        avg.rating = avg.rating_avg.astype(int)
        no_rat = df_clk_rat_star.loc[df_clk_rat_star.rating == 1, ['user_id','session_id','item_id']]
        rat_filled = pd.merge(no_rat, avg, how='left', on=['user_id','session_id']).fillna(0)
        rat_filled.rating = rat_filled.rating_avg.astype(int)
        df_clk_rat_star_rat_filled = pd.merge(df_clk_rat_star_filled, rat_filled, how='left',
                                              on = ['user_id','session_id','item_id'])
        for t in zip(df_clk_rat_star_rat_filled.rating, df_clk_rat_star_rat_filled.rating_avg,
                      df_clk_rat_star_rat_filled.index):
            if t[0] == 1:
                df_clk_rat_star_rat_filled.at[t[2], 'rating'] = t[1]
        df_clk_rat_star_rat_filled = df_clk_rat_star_rat_filled.drop('rating_avg',1)

        # add feature column
        new_col = []
        df_clk_rat_star_rat_filled.rating = df_clk_rat_star_rat_filled.rating.astype(int)
        df_clk_rat_star_rat_filled.stars = df_clk_rat_star_rat_filled.stars.astype(int)
        df_clk_rat_star_rat_filled.price = df_clk_rat_star_rat_filled.price.astype(int)

        for t in tqdm(zip(df_clk_rat_star_rat_filled.rating, df_clk_rat_star_rat_filled.stars,
                          df_clk_rat_star_rat_filled.price)):
            new_col.append((1.5*t[0]+t[1])/t[2])
        df_clk_rat_star_rat_filled['price_quality'] = new_col
        final_feature = df_clk_rat_star_rat_filled[['user_id','session_id','item_id','price_quality']]

        return final_feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = PriceQuality(mode=mode, cluster=cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
