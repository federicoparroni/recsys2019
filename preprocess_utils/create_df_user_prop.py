import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import os
os.chdir("../")
print(os.getcwd())

"""
creates train and test dataframe that can be used for classification,
starting from train_df and test_df of the specified cluster and mode

the resulting dataframe have structure:
user_id | session_id | accomodation_id | feature_1 | ....  | feature_n | label

the accomodation_ids are the ones showing up in the impressions
label is 1 in case the accomodation is the one clicked in the clickout
"""


def build_user_prop(mode, cluster='no_cluster'):

    def func(x):

        y = x[(x['action_type'] == 'clickout item')]

        # features
        features = {'avg price interacted items': 0, 'avg position asc price interacted items': 0,
                    'session avg length': 0, 'session avg steps': 0, 'session num': 0,
                    'mobile perc': 0, 'tablet perc': 0, 'desktop perc': 0, 'filters_when_clickout': '',
                    'num change of sort order session': 0, 'num clickout item session': 0,
                    'num filter selection session': 0, 'num interaction item deals session': 0,
                    'num interaction item image session': 0, 'num interaction item info session': 0,
                    'num interaction item rating session': 0, 'num search for destination session': 0,
                    'num search for item session': 0, 'num search for poi session': 0}

        # Compute avg lenght of session in seconds (OSS: not considering session ending at last clickout!)
        session_grouped = x.groupby("session_id")
        get_lenght_sum = 0
        for name, group in session_grouped:
            get_lenght_sum += int(group.tail(1)['timestamp'].values[0]) - int(group.head(1)['timestamp'].values[0])

        # Compure avg steps in a session (OSS: not considering session ending at last clickout!)
        user_sessions = set(x['session_id'].values)
        features['session avg steps'] = round(len(x) / len(user_sessions) , 2)

        features['session num'] = len(user_sessions)

        features['session_avg_length'] = round(get_lenght_sum / len(user_sessions), 2)

        # Computing types of non_numeric actions performed by that user in the past
        actions = list(x['action_type'].values)

        for ind in range(len(actions)):
            if ('num ' + actions[ind] + ' session') in features:
                features['num ' + actions[ind] + ' session'] += 1

        # Remove duplicates:
        x.drop(['timestamp', 'step'], axis=1, inplace=True)
        x = x.drop_duplicates()

        if len(y) > 0:

            # Builld a record of interacted price of items only when available:
            impressions_prices_available = y[y['impressions'] != None][["impressions", "prices"]].drop_duplicates()
            # [13, 43, 4352, 543, 345, 3523] impressions
            # [45, 34, 54, 54, 56, 54] prices
            # -> [(13,45), (43,34), ...]
            # Then create dict
            # {13: 45, 43: 34, ... }

            tuples_impr_prices = []
            tuples_impr_price_pos_asc = []
            for i in impressions_prices_available.index:
                impr = impressions_prices_available.at[i, 'impressions'].split('|')
                prices = impressions_prices_available.at[i, 'prices'].split('|')
                tuples_impr_prices += list(zip(impr, prices))

                sorted(tuples_impr_prices, key=lambda x: x[1])

                tuples_impr_price_pos_asc += list(zip(impr, list(range(1, len(tuples_impr_prices) + 1))))

            tuples_impr_prices = list(set(tuples_impr_prices))
            dict_impr_price = dict(tuples_impr_prices)

            # Create dict for getting position wrt clicked impression based on cheapest item
            tuples_impr_price_pos_asc = list(set(tuples_impr_price_pos_asc))
            dict_impr_price_pos = dict(tuples_impr_price_pos_asc)

            sum_price = 0
            sum_pos_price = 0
            count_interacted = 0

            # IMPORTANT: I decided to consider impressions and clickouts distinctively.
            # If an impression is also clicked, that price counts double
            df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][
                ["reference", "impressions"]].drop_duplicates()

            for reference in df_only_numeric.reference:
                if reference in dict_impr_price.keys():
                    sum_price += int(dict_impr_price[reference])
                    sum_pos_price += int(dict_impr_price_pos[reference])
                    count_interacted += 1

            if count_interacted > 0:
                features['avg price interacted items'] = round(sum_price / count_interacted , 2)
                features['avg position asc price interacted items'] = round(sum_pos_price / count_interacted , 2)

            # Device percentages features
            tot_clks = len(y)
            features['mobile perc'] = round(y[y.device == "mobile"].shape[0] / tot_clks , 2)
            features['tablet perc'] = round(y[y.device == "tablet"].shape[0] / tot_clks , 2)
            features['desktop perc'] = round(y[y.device == "desktop"].shape[0] / tot_clks , 2)

            # Getting used filters during past clickouts (except during clickout to predict!), then they will be one_hotted
            y_filters = y[(y.current_filters != None) & (y.reference != None)]
            for i in y_filters.index:
                features['filters_when_clickout'] += str(y_filters.at[i, 'current_filters']) + "|"

        return pd.DataFrame.from_records([features])

    def construct_features(df):
        dataset = df.groupby(['user_id']).progress_apply(func)

        one_hot = dataset['filters_when_clickout'].astype(str).str.get_dummies()
        missing = poss_filters - set(one_hot.columns)

        to_drop = set(one_hot.columns) - poss_filters
        for e in missing:
            one_hot[e] = 0
        for e in to_drop:
            one_hot = one_hot.drop([e], axis=1)
        dataset = dataset.drop(['filters_when_clickout'], axis=1)
        dataset = dataset.join(one_hot)

        return dataset

    def get_user_favorite_filters(full_df, users):
        """
        I want a structure that for every user in the train gives
        an one_hot_encoded structures for all possible parameters of hotels clicked by that user
        ex. parameter: 3 Stars
        """

        # get clickout of train and merge metadata of the hotel
        train_df = full_df[full_df["user_id"].isin(users)]
        train_df = train_df[(train_df["action_type"] == "clickout item") & (pd.to_numeric(train_df['reference'], errors='coerce').notnull())]

        train_df.drop(
            ["session_id", "timestamp", "step", "action_type", "platform", "city", "device", "current_filters",
             "impressions", "prices"], axis=1, inplace=True)

        # Merge & eliminate column
        metatadata_one_hot = data.get_accomodations_one_hot()

        train_df['reference'] = train_df['reference'].astype(int)
        metatadata_one_hot['item_id'] = metatadata_one_hot['item_id'].astype(int)
        train_df = pd.merge(train_df, metatadata_one_hot, how='outer', left_on='reference', right_on='item_id')

        train_df = train_df.drop(["reference", "item_id"], axis=1)

        print("Finishing binaryzing, now summing...")

        out_df = train_df.groupby('user_id')[train_df.columns[2:]].sum()
        return out_df

    # Start trying to compute dataset
    train = data.train_df(mode=mode, cluster=cluster)

    test = data.test_df(mode=mode, cluster=cluster)
    target_indices = data.target_indices(mode=mode, cluster=cluster)
    target_user_id = test.loc[target_indices]['user_id'].values

    full = pd.concat([train, test])
    del train
    del test

    poss_filters = []
    for f in full[~full['current_filters'].isnull()]['current_filters'].values:
        poss_filters += f.split('|')
    poss_filters = set(poss_filters)

    user_fav_filters = get_user_favorite_filters(full, target_user_id)

    #Add suffix in order to distinguish hotel properties from user filters
    user_fav_filters.columns = [str(col) + '_hotelProp' for col in user_fav_filters.columns]

    user_fav_filters.reset_index(inplace=True)

    # build in chunk
    count_chunk = 0
    chunk_size = 100000

    print("{}: Started chunk processing".format("Build Dataset Classification"))
    groups = full.groupby(np.arange(len(full)) // chunk_size)
    for idxs, gr in groups:
        features = construct_features(gr)

        features.reset_index(inplace=True)

        outcome = pd.merge(features, user_fav_filters, how='outer', left_on="user_id", right_on="user_id")
        outcome.drop(["level_1", outcome.columns.values[-1]], axis=1, inplace=True)
        outcome.fillna(0, inplace=True)

        if count_chunk == 0:
            outcome.to_csv(
                'dataset/preprocessed/{}/{}/user_properties.csv'.format(cluster, mode))
        else:
            with open('dataset/preprocessed/{}/{}/user_properties.csv'.format(cluster, mode), 'a') as f:
                outcome.to_csv(f, header=False)

        count_chunk += 1
        print('chunk {} over {} completed'.format(count_chunk, len(groups)))


if __name__ == "__main__":
    build_user_prop(mode='small', cluster="no_cluster")
