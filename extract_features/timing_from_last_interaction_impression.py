from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions

import os
#os.chdir("../")
#print(os.getcwd())

class TimingFromLastInteractionImpression(FeatureBase):

    """
    how much time is elapsed and how many steps are passed from the last time a user
    interacted with an impression
    | user_id | session_id | item_id |step_from_last_interaction|timestamp_from_last_interaction
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'timing_from_last_interaction_impression'
        super(TimingFromLastInteractionImpression, self).__init__(
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

        impr_features = {}
        impr_feature = []
        for i in tqdm(sorted(idxs_numeric_reference)):
            if i == last_click:
                impressions = list(map(int, temp.at[i, 'impressions'].split('|')))
                click_timestamp = temp.at[i, 'timestamp']
                click_step = temp.at[i, 'step']
                for impr in impressions:
                    if impr not in impr_features:
                        impr_feature.append({'num_interactions_impr': 0, 'step_from_last_interaction': -1,
                                             'timestamp_from_last_interaction': -1,
                                             'last_action_type_with_impr': 'None'})
                    else:
                        impr_features[impr]['timestamp_from_last_interaction'] = click_timestamp - impr_features[impr][
                            'timestamp_from_last_interaction']
                        impr_features[impr]['step_from_last_interaction'] = click_step - impr_features[impr][
                            'step_from_last_interaction']
                        impr_feature.append(impr_features[impr])
                impr_features = {}
                count += 1
                if count < len(idxs_click):
                    last_click = idxs_click[count]
                continue
            ref = int(temp.at[i, 'reference'])
            if ref in impr_features:
                impr_features[ref]['num_interactions_impr'] += 1
                impr_features[ref]['step_from_last_interaction'] = df.at[i, 'step']
                impr_features[ref]['timestamp_from_last_interaction'] = df.at[i, 'timestamp']
                impr_features[ref]['last_action_type_with_impr'] = df.at[i, 'action_type']
            else:
                impr_features[ref] = {'num_interactions_impr': 1, 'step_from_last_interaction': df.at[i, 'step'],
                                      'timestamp_from_last_interaction': df.at[i, 'timestamp'],
                                      'last_action_type_with_impr': df.at[i, 'action_type']}

        final_df = expand_impressions(temp[['user_id', 'session_id', 'impressions']].loc[idxs_click])
        print(len(final_df))
        print(len(impr_feature))
        final_df['dict'] = impr_feature

        features_df = pd.DataFrame(final_df.progress_apply(lambda x: tuple(x['dict'].values()), axis=1).tolist(),
                                   columns=list(final_df.iloc[0].dict.keys()))
        final_df_ = pd.concat([final_df, features_df], axis=1).drop('dict', axis=1)
        final_df_ = final_df_.drop(['num_interactions_impr', 'last_action_type_with_impr'], axis=1)
        return final_df_

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = TimingFromLastInteractionImpression(mode=mode, cluster=cluster)
    c.save_feature()

