from extract_features.feature_base import FeatureBase
import data
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()
from preprocess_utils.last_clickout_indices import expand_impressions
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances

class User2ItemOld(FeatureBase):


    def __init__(self, mode, cluster='no_cluster'):
        name = f'user_2_item_old'
        super(User2ItemOld, self).__init__(name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def create_user_feature_dict(df):

            idxs_clicks = find_last_clickout_indices(df)
            df = df.drop(idxs_clicks)

            # retrieve the icm
            icm = data.accomodations_one_hot()

            # filter on the columns of interests
            temp_df = df[['user_id', 'session_id', 'reference']].dropna()

            # mantain only the rows with numeric reference
            temp_df = temp_df[temp_df['reference'].str.isnumeric()]
            temp_df = temp_df.drop_duplicates()

            # retrieve user_sess and item associated to it from the dataframe
            users_idxs = temp_df.to_dict('list')['user_id']
            sessions_idxs = temp_df.to_dict('list')['session_id']
            users_sess = list(zip(users_idxs, sessions_idxs))
            items_idxs = list(map(int, temp_df.to_dict('list')['reference']))

            # create a diction with keys tuples like ('user_id', 'session_id') and as value the array representing
            # the user as an house summing up all the features of the houses with wich he has interacted
            # during the session
            count = 0
            user_session_dict = {}
            for user_session in tqdm(users_sess):
                user_items = icm.loc[items_idxs[count]].values
                if user_session in user_session_dict:
                    user_session_dict[user_session] += user_items
                else:
                    user_session_dict[user_session] = user_items
                count += 1

            return user_session_dict

        def retrieve_pd_dataframe_score(df):

            icm = data.accomodations_one_hot().sort_index()

            sess_user_dict = create_user_feature_dict(df)
            idxs_clicks = find_last_clickout_indices(df)

            scores = []
            # iterate on the index of the target clicks and create for each iteration a tuple to be appended on the final list
            print('computing the distances...')
            for idx in tqdm(idxs_clicks):

                # retrieve the user sess and impressions of the click
                user = df.at[idx, 'user_id']
                sess = df.at[idx, 'session_id']
                impressions = list(map(int, df.at[idx, 'impressions'].split('|')))

                # retrieve the impression of the user-sess pair if it isn't in the dictionary means
                # that there weren't numeric actions in the session so initialize it with an empty vector
                us_tuple = (user, sess)

                if us_tuple in sess_user_dict:
                    user_feature_vec = sess_user_dict[(user, sess)]
                else:
                    user_feature_vec = np.zeros(icm.shape[1])

                # retrieve the features of the impression selected
                features_imp = icm.loc[impressions].values

                # create the various version of the user vector CLIPPED, TRESHOLDED
                clipped_user_feature_vec = np.clip(user_feature_vec,0,1)

                tresholded_user_feature_vec = user_feature_vec.copy()
                if np.sum(user_feature_vec) > 0:
                    treshold_limit = np.mean(user_feature_vec[user_feature_vec > 0])
                    tresholded_user_feature_vec[tresholded_user_feature_vec<treshold_limit]=0

                # compute the distance between the two vectors
                _scores_manhattan = manhattan_distances(user_feature_vec.reshape(1, -1), features_imp)
                _scores_cosine = cosine_similarity(user_feature_vec.reshape(1, -1), features_imp)

                # create and append a tuple on the final list
                for i in range(len(impressions)):
                    scores.append((user, sess, impressions[i],
                                   _scores_cosine[0][i], _scores_manhattan[0][i]))
            return pd.DataFrame(scores, columns=['user_id', 'session_id', 'item_id',
                                                 'scores_cosine', 'scores_manhatthan'])

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)

        # concat the train and test dataframe and retrieve the indices of the last clickouts of the sessions
        df = pd.concat([train, test]).sort_values(
            ['user_id', 'session_id', 'timestamp', 'step']).reset_index(drop=True)

        return retrieve_pd_dataframe_score(df)


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = User2ItemOld(mode=mode, cluster=cluster)
    c.save_feature()
