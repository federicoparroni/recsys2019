from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

class UserFeature(FeatureBase):
    def __init__(self, mode, cluster='no_cluster'):
        name = 'user_feature'
        super(UserFeature, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        ######### READING DATA
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])


        ######## SOME PREPROCESS + SECONDARY DATA STRUCTURE TO SPEED UP PERFOMANCES
        clickout_indices = find(df)
        clickout_df = df.loc[clickout_indices]
        clickout_sessions = list(clickout_df.session_id)
        session_to_impressions = dict()
        user_to_sessions = dict()
        session_to_timestamp = dict()
        for t in tqdm(zip(clickout_df.session_id, clickout_df.impressions, clickout_df.user_id, clickout_df.timestamp)):
            if t[2] not in user_to_sessions:
                user_to_sessions[t[2]] = list()
            user_to_sessions[t[2]] += [t[0]]
            session_to_impressions[t[0]] = list(map(int, t[1].split("|")))
            session_to_timestamp[t[0]] = t[3]

        # Cleaning df from clickout sessions and not numeric reference (i.e Castellammare di Stabbia, NA) and from users
        # which are not in the test set
        clean_df = df[(~df.session_id.isin(clickout_sessions)) & (df.reference.apply(lambda x: type(x) == str and x.isdigit())) \
                        & (df.user_id.isin(user_to_sessions.keys()))]
        clean_df["reference"] = pd.to_numeric(clean_df["reference"])
        grouped = clean_df.groupby("user_id")
        session_to_df = dict()
        for name, group in tqdm(grouped, desc="Scanning users and create enriched sessions dataframe"):
            group = group.sort_values("timestamp")
            sessions = user_to_sessions[name]
            #Attach to each session a small df containing only the rows useful for the computation of the feature
            for s in sessions:
                imps = session_to_impressions[s]
                temp = group[group.reference.isin(imps)]
                session_to_df[s] = temp

        print(len(session_to_df))

        #### FEATURE KERNEL

        # Action mapping on indices
        # 0 -> time_from_last_interaction
        # Action <-> index of the array
        time_last_interaction_past = 0
        action_dict_past = {
            "search for item" : 1,
            "interaction item image" : 2,
            "interaction item info" : 3,
            "interaction item deals" : 4,
            "interaction item rating" : 5,
            "clickout item" : 6}
        time_first_interaction_future = 7
        action_dict_future = {
            "search for item" : 8,
            "interaction item image" : 9,
            "interaction item info" : 10,
            "interaction item deals" :11,
            "interaction item rating" : 12,
            "clickout item" : 13}
        imp_to_actions = dict()
        session_to_feature = dict()
        for k, v in tqdm(session_to_timestamp.items(), desc="Scanning sessions to generate feature"):
            # if not, we don't have any information from past or future, from that user, for that impressions
            if k in session_to_df:
                temp = session_to_df[k]
                past = temp[temp.timestamp <= v].sort_values("timestamp")
                future = temp[temp.timestamp > v].sort_values("timestamp", ascending=False)
                imps = session_to_impressions[k]
                imp_to_actions = dict()
                for i in imps:
                    # + 2 due to "time_from_last_interaction", both past and future
                    imp_to_actions[i] = np.zeros(len(action_dict_past) + len(action_dict_future) + 2)
                    imp_to_actions[i][time_last_interaction_past] = -1
                    imp_to_actions[i][time_first_interaction_future] = -1
                for t in zip(past.reference, past.action_type, past.timestamp):
                    imp = t[0]
                    action_index = action_dict_past[t[1]]
                    imp_to_actions[imp][time_last_interaction_past] = v - t[2]
                    imp_to_actions[imp][action_index] += 1
                for t in zip(future.reference, future.action_type, future.timestamp):
                    imp = t[0]
                    action_index = action_dict_future[t[1]]
                    imp_to_actions[imp][time_first_interaction_future] = t[2] - v
                    imp_to_actions[imp][action_index] += 1
                session_to_feature[k] = imp_to_actions

        #### UNROLLING DICT TO DATAFRAME
        lines = list()
        for k, v in tqdm(session_to_feature.items(), desc="Dicts to dataframe"):
            for imp, feature in v.items():
                lines.append([k] + [imp] + list(feature))
        new_df = pd.DataFrame(lines, columns=["session_id", "item_id", "time_last_past_interaction", "search_for_item_past",
                                              "interaction_item_image_past", "interaction_item_info_past", "interaction_item_deals_past",
                                              "interaction_item_rating_past", "clickout_item_past", "time_first_future_interaction", "search_for_item_future",
                                               "interaction_item_image_future", "interaction_item_info_future", "interaction_item_deals_future",
                                              "interaction_item_rating_future", "clickout_item_future"])

        #### MERGING WITH MAIN DATAFRAME

        clickout_df = clickout_df[['user_id', 'session_id', 'impressions']]
        clk_expanded = expand_impressions(clickout_df)
        print("Temp feature (only rows not null) shape: {}".format(new_df.shape))
        print("Expanded dataframe shape: {}".format(clk_expanded.shape))
        feature = pd.merge(clk_expanded, new_df, how="left")
        feature[["search_for_item_past","interaction_item_image_past", "interaction_item_info_past", "interaction_item_deals_past",
                "interaction_item_rating_past", "clickout_item_past", "search_for_item_future",
                "interaction_item_image_future", "interaction_item_info_future", "interaction_item_deals_future",
                "interaction_item_rating_future", "clickout_item_future"]] = \
            feature[["search_for_item_past","interaction_item_image_past", "interaction_item_info_past", "interaction_item_deals_past",
                "interaction_item_rating_past", "clickout_item_past", "search_for_item_future",
                "interaction_item_image_future", "interaction_item_info_future", "interaction_item_deals_future",
                "interaction_item_rating_future", "clickout_item_future"]].fillna(value=0)
        feature.replace(-1, np.nan)
        print("Final feature shape: {}".format(feature.shape))
        return feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = UserFeature(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()
