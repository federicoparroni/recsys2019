from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions

class StatisticsPosInteracted(FeatureBase):

    """
    | user_id | session_id | min_pos_interacted | first_pos_interacted | num_interacted_impressions | percentage_interacted_impressions
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'statistics_pos_interacted'
        super(StatisticsPosInteracted, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        temp = df.fillna('0')
        idxs_click = sorted(find_last_clickout_indices(temp))
        idxs_numeric_reference = temp[temp['reference'].str.isnumeric() == True].index

        count = 0
        last_click = idxs_click[0]

        sess_features_dict = {}
        sess_feature = []

        for i in tqdm(sorted(idxs_numeric_reference)):
            n_user = df.at[last_click, 'user_id']
            n_sess = df.at[last_click, 'session_id']
            if i == last_click:
                impressions = list(map(int, temp.at[i, 'impressions'].split('|')))

                impressions_len = len(impressions)

                num_interacted_impressions = 0
                impression_interacted = set()
                positions_interacted = []

                tuples = sorted(list(sess_features_dict.items()), key=lambda t: t[0])

                min_pos = 26
                max_pos = -1
                first_pos = None
                last_pos = None
                mean_pos_interacted = -1

                for t in tuples:
                    try:
                        index = impressions.index(t[1]) + 1
                        if first_pos is None:
                            first_pos = index
                        last_pos = index
                        if index < min_pos:
                            min_pos = index
                        if index > max_pos:
                            max_pos = index
                        num_interacted_impressions += 1
                        impression_interacted.add(t[1])
                        positions_interacted.append(index)
                    except ValueError:
                        pass
                if impressions_len > 0:
                    percentage_interacted_impression = len(impression_interacted) / impressions_len
                else:
                    percentage_interacted_impression = 1

                if num_interacted_impressions == 0:
                    min_pos = -1
                    max_pos = -1
                    first_pos = -1
                    last_pos = -1

                if num_interacted_impressions > 0:
                    mean_pos_interacted = sum(positions_interacted)/len(positions_interacted)

                f_d = {
                    'mean_pos_interacted': mean_pos_interacted,
                    'min_pos_interacted': min_pos,
                    'max_pos_interacted': max_pos,
                    'first_pos_interacted': first_pos,
                    'last_pos_interacted': last_pos,
                    'num_interacted_impressions': num_interacted_impressions,
                    'percentage_interacted_impressions': percentage_interacted_impression
                }
                sess_feature.append(f_d)
                sess_features_dict = {}
                count += 1
                if count < len(idxs_click):
                    last_click = idxs_click[count]
                continue
            
            if (temp.at[i, 'user_id'] == n_user) and (temp.at[i, 'session_id'] == n_sess):
                ref = int(temp.at[i, 'reference'])
                step_interaction = temp.at[i, 'step']
                sess_features_dict[step_interaction] = ref

        final_df = temp[['user_id', 'session_id']].loc[idxs_click]
        final_df['dict'] = sess_feature

        features_df = pd.DataFrame(final_df.progress_apply(lambda x: tuple(x['dict'].values()), axis=1).tolist(),
                                   columns=list(final_df.iloc[0].dict.keys()))
        final_df_ = pd.merge(final_df.drop('dict', axis=1).reset_index(drop=True).reset_index(),
                             features_df.reset_index()).drop('index', axis=1)
        return final_df_.drop(['mean_pos_interacted', 'max_pos_interacted', 'last_pos_interacted'], axis=1)

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = StatisticsPosInteracted(mode=mode, cluster=cluster)
    c.save_feature()

