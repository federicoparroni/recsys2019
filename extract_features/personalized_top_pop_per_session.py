from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

class PersonalizedTopPopPerSession(FeatureBase):

    """
    This class computes the top popular per clickout in this way:
    - counts all the times each accomodation is clicked in bot train and test
        without masking the last clickout, removing multiple clicks from the same
        user/session
    - for each clickout session, subtracts 1 from the popularity of an impression
        in the impression list if it is the one clicked

    Example:

    Overall popularity:
    items  | pop
    ------------
    item_1 | 2
    item_2 | 3
    ...

    Clickout session in which the clicked item is item_2:
    impress | pop
    -------------
    item_4  | 1
    item_2  | 2  <---- this was 3 overall, but one of the three clickout comes from this session!
    ...

    The feature format is:

    user_id | session_id | item_id | personalized_popularity_per_session

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'personalized_top_pop_per_session'
        super(PersonalizedTopPopPerSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','reference','action_type','impressions']]
        reference_rows = df[(df.reference.str.isnumeric() == True) & (df.action_type == 'clickout item')]
        reference_rows = reference_rows.drop_duplicates(['user_id','session_id','reference'])

        df_item_clicks = (
            reference_rows
            .groupby(["reference"])
            .size()
            .reset_index(name="n_interactions_per_item")
        )
        df_item_clicks = df_item_clicks.rename(columns={'reference':'item_id'})
        df_item_clicks['item_id'] = df_item_clicks['item_id'].astype(int)
        #df_item_clicks

        clk_expanded = expand_impressions(clickout_rows)
        final_feature = pd.merge(clk_expanded, df_item_clicks, how='left', on=['item_id']).fillna(0)
        final_feature.n_interactions_per_item = final_feature.n_interactions_per_item.astype(int)
        final_feature = final_feature.drop(['index'], axis=1)

        final_feature.reference = final_feature.reference.astype(int)
        new_column = []
        for t in zip(final_feature.item_id, final_feature.reference, final_feature.n_interactions_per_item):
            if t[0] == t[1]:
                new_column.append(int(t[2]-1))
            else:
                new_column.append(int(t[2]))
        final_feature['personalized_popularity_per_session'] = new_column

        final_feature_reduced = final_feature[['user_id','session_id','item_id','personalized_popularity_per_session']]

        return final_feature_reduced

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = PersonalizedTopPopPerSession(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
