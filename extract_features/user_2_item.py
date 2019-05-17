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
from sklearn.metrics import jaccard_similarity_score

class User2Item(FeatureBase):


    def __init__(self, mode, cluster='no_cluster', repeated_reference=False):
        print('do you want repeated reference? (y or n)')
        user_inp = input()
        if user_inp == 'y':
            self.repeated_reference=True
        else:
            self.repeated_reference = False
        print('which distance do you want to use?:\n'
              '1) manhattan_distances\n'
              '2) cosine_similarity\n'
              '3) jaccard_similarity\n')
        user_inp = input()
        if user_inp == '1':
            self.distance = 'manhattan_distances'
        elif user_inp == '2':
            self.distance = 'cosine_similarity'
        else:
            self.distance = 'jaccard_similarity'
        name = f'user_2_item_rr_{repeated_reference}_dist_{self.distance}'
        super(User2Item, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def create_user_feature_dict(df):

            # retrieve the icm
            icm = data.accomodations_one_hot()

            # filter on the columns of interests
            temp_df = df[['user_id', 'session_id', 'action_type', 'reference']]

            # create a diction with keys tuples like ('user_id', 'session_id') and as value the array representing
            # the user as an house summing up all the features of the houses with wich he has interacted
            # during the session
            sess_user_dict = {}

            # create a mask that will give us only the row of the dataframe with a numeric reference
            mask = temp_df["reference"].apply(lambda x: type(x) == str and x.isdigit())
            temp_df = temp_df[mask]

            # fill the dictionary
            groups = temp_df.groupby(['user_id', 'session_id'])
            for name, group in tqdm(groups):

                # initialize user-feature vector with zeros
                user_features_vec = np.zeros(icm.shape[1])

                # cut the session to the last clickout
                click = group[group['action_type'] == 'clickout item']
                if len(click > 1):
                    last_click_index = click.tail(1)
                    head_index = group.head(1).index
                    group = group.loc[head_index.values[0]:last_click_index.index.values[0] - 1]

                # retrieve the references of the session
                if not self.repeated_reference:
                    ref = list(map(int, group.reference.unique()))
                else:
                    ref = list(map(int, group.reference))

                    # for each reference access the icm and sum the features to the user-feature_vec
                for r in ref:
                    user_features_vec += icm.loc[r].values

                # insert in the dictionary the user_feature_vec
                sess_user_dict[name] = user_features_vec
            return sess_user_dict

        def retrieve_pd_dataframe_score(df):

            icm = data.accomodations_one_hot()

            sess_user_dict = create_user_feature_dict(df)
            idxs_clicks = find_last_clickout_indices(df, sort=False)

            final_list = []

            # iterate on the index of the target clicks and create for each iteration a tuple to be appended on the final list
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

                # TODO _THRESHOLD will be moved as parameter

                # retrieve the features of the impression selected
                features_imp = icm.loc[impressions].values

                # compute the distance between the two vectors

                if self.distance == 'manhattan_distances':
                    scores = manhattan_distances(user_feature_vec.reshape(1, -1), features_imp)
                elif self.distance == 'cosine_similarity':
                    scores = cosine_similarity(user_feature_vec.reshape(1, -1), features_imp)
                else:
                    # need to perform a clip to one since for the jaccard similarity we need labels
                    scores = []
                    # Jaccard similarity doesnt support matrix product we will do it impression by impression
                    for num_impr in range(len(impressions)):
                        score = jaccard_similarity_score(np.clip(user_feature_vec, 1, 0), features_imp[num_impr])
                        scores.append(score)
                    scores = np.array(scores).reshape(1, -1)

                # create and append a tuple on the final list
                for i in range(len(impressions)):
                    final_list.append((user, sess, impressions[i], scores[0][i]))

            return pd.DataFrame(final_list, columns=['user_id', 'session_id', 'item_id', f'sim_score_{self.distance}'])

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)

        # concat the train and test dataframe and retrieve the indices of the last clickouts of the sessions
        df = pd.concat([train, test]).sort_values(
            ['user_id', 'session_id', 'timestamp', 'step']).reset_index(drop=True)

        return retrieve_pd_dataframe_score(df)


if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = User2Item(mode=mode, cluster='no_cluster')
    c.save_feature()
