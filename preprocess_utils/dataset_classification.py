import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from utils.check_folder import check_folder
from multiprocessing import Process, Queue

"""
creates train and test dataframe that can be used for classification,
starting from train_df and test_df of the specified cluster and mode

the resulting dataframe have structure:
user_id | session_id | accomodation_id | feature_1 | ....  | feature_n | label

the accomodation_ids are the ones showing up in the impressions
label is 1 in case the accomodation is the one clicked in the clickout
"""

import os
os.chdir("../")
print(os.getcwd())

def build_dataset(mode, cluster='no_cluster', algo='xgboost'):

    # build the onehot of accomodations attributes
    def one_hot_of_accomodation(accomodations_df):
        accomodations_df.properties = accomodations_df.properties.progress_apply(
            lambda x: x.split('|') if isinstance(x, str) else x)
        accomodations_df.fillna(value='', inplace=True)
        mlb = MultiLabelBinarizer()
        one_hot_features = mlb.fit_transform(accomodations_df.properties)
        one_hot_accomodations_df = pd.DataFrame(
            one_hot_features, columns=mlb.classes_)
        one_hot_accomodations_df.columns = [
            'accomodation feature ' + str(col) for col in one_hot_accomodations_df.columns]
        attributes_df = pd.concat([accomodations_df.drop(
            'properties', axis=1), one_hot_accomodations_df], axis=1)
        return attributes_df
        # return accomodations_df

    def build_popularity(df):
        popularity = {}
        df = df[(df['action_type'] == 'clickout item')
                & (~df['reference'].isnull())]
        clicked_references = list(map(int, list(df['reference'].values)))

        ## FREQUENCY EDIT - in case of presence of 'frequence' column in dataset
        if 'frequence' in df.columns.values:
            frequence = list(map(int, list(df['frequence'].values)))
        else:
            frequence = [1] * len(df)

        for i in tqdm(range(len(clicked_references))):
            e = clicked_references[i]
            f = frequence[i]
            if int(e) in popularity:
                popularity[int(e)] += int(f)
            else:
                popularity[int(e)] = int(f)
        return popularity

    def get_price_info_and_interaction_position(x, y):
        """
        Getting avg price of interacted items and average price position inside a given session
        :param x:
        :param y:
        :return:
        """
        impressions_pos_available = y[y['impressions'] != None][['impressions', 'prices']].drop_duplicates()

        # [13, 43, 4352, 543, 345, 3523] impressions
        # [45, 34, 54, 54, 56, 54] prices
        # -> [(13,45), (43,34), ...]
        # Then create dict
        # {13: 45, 43: 34, ... }

        tuples_impr_prices = []
        tuples_impr_price_pos_asc = []

        # [13, 43, 4352, 543, 345, 3523] impressions
        # Then create dict impression-position
        # {13: 1, 43: 2, ... }
        tuples_impr_pos = []

        for i in impressions_pos_available.index:
            impr = impressions_pos_available.at[i, 'impressions'].split('|')
            prices = impressions_pos_available.at[i, 'prices'].split('|')
            tuples_impr_prices += list(zip(impr, prices))

            tuples_impr_pos += [(impr[idx], idx + 1) for idx in range(len(impr))]

            sorted(tuples_impr_prices, key=lambda x: x[1])
            tuples_impr_price_pos_asc += list(zip(impr, list(range(1, len(tuples_impr_prices) + 1))))

        tuples_impr_prices = list(set(tuples_impr_prices))
        dict_impr_price = dict(tuples_impr_prices)

        dict_impr_pos = dict(list(set(tuples_impr_pos)))

        sum_pos_impr = 0
        count_interacted_pos_impr = 0

        # Create dict for getting position wrt clicked impression based on cheapest item
        tuples_impr_price_pos_asc = list(set(tuples_impr_price_pos_asc))
        dict_impr_price_pos = dict(tuples_impr_price_pos_asc)

        sum_price = 0
        sum_pos_price = 0
        count_interacted = 0

        # IMPORTANT: I decided to consider impressions and clickouts distinctively.
        # If an impression is also clicked, that price counts double

        #considering reference, impressions and action type as a row, I can distinguish from clickouts and impressions dropping duplicates
        df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][
            ["reference", "impressions", "action_type"]].drop_duplicates()

        # Not considering last clickout in the train sessions
        clks_num_reference = df_only_numeric[df_only_numeric['action_type'] == 'clickout item']
        if len(y) > 0 and len(clks_num_reference) == len(y):  # is it a train session?
            idx_last_clk = y.tail(1).index.values[0]
            df_only_numeric = df_only_numeric.drop(idx_last_clk)

        for i in df_only_numeric.index:
            reference = df_only_numeric.at[i, 'reference']

            if reference in dict_impr_price.keys():

                sum_pos_impr += int(dict_impr_pos[reference])
                count_interacted_pos_impr += 1
                count_interacted += 1

                sum_price += int(dict_impr_price[reference])
                sum_pos_price += int(dict_impr_price_pos[reference])

        mean_pos = -1
        pos_last_reference = -1

        mean_cheap_position = -1
        mean_price_interacted = -1

        if count_interacted > 0:
            mean_cheap_position = round(sum_pos_price / count_interacted, 2)
            mean_price_interacted = round(sum_price / count_interacted, 2)

            mean_pos = round(sum_pos_impr / count_interacted_pos_impr, 2)
            last_reference = df_only_numeric.tail(1).reference.values[0]

            # Saving the impressions appearing in the last clickout (they will be used to get the 'pos_last_reference'
            impressions_last_clickout = y.tail(1).impressions.values[0].split('|')
            if last_reference in impressions_last_clickout:
                pos_last_reference = impressions_last_clickout.index(last_reference) + 1

        return mean_cheap_position, mean_price_interacted, mean_pos, pos_last_reference


    def get_frenzy_and_avg_time_per_step(x):
        if len(x) > 1:
            session_actions_num = int(x.tail(1).step)

            time_length = int(x.tail(1).timestamp) - int(x.head(1).timestamp)

            mean_time_per_step = round(time_length / session_actions_num, 2)

            var = 0
            prev_tm = 0

            for i in x.index:
                curr_tm = int(x.at[i, 'timestamp'])

                if prev_tm == 0:
                    prev_tm = curr_tm
                else:
                    var += (mean_time_per_step - (curr_tm - prev_tm)) ** 2
                    prev_tm = curr_tm

            var = round((var / session_actions_num) ** 0.5, 2)
        else:
            var = -1
            mean_time_per_step = -1

        return mean_time_per_step, var

    def func(x):
        y = x[(x['action_type'] == 'clickout item')]
        if len(y) > 0:
            clk = y.tail(1)
            head_index = x.head(1).index
            impr = clk['impressions'].values[0].split('|')

            #!! considering only the past for all non-test sessions! !!
            x = x.loc[head_index.values[0]:clk.index.values[0]-1]

            # features
            features = {'label': [], 'times_impression_appeared': [], 'time_elapsed_from_last_time_impression_appeared': [], 'impression_position': [],
                        'steps_from_last_time_impression_appeared': [], 'kind_action_reference_appeared_last_time': [], 'price': [], 'price_position': [],
                        'item_id': [], 'popularity': [], 'impression_position_wrt_last_interaction': [], 'impression_position_wrt_second_last_interaction': [], 'clickout_item_session_ref_this_impr': list(np.zeros(len(impr), dtype=np.int)),
                        'interaction_item_deals_session_ref_this_impr': list(np.zeros(len(impr), dtype=np.int)), 'interaction_item_image_session_ref_this_impr': list(np.zeros(len(impr), dtype=np.int)),
                        'interaction_item_info_session_ref_this_impr': list(np.zeros(len(impr), dtype=np.int)), 'interaction_item_rating_session_ref_this_impr': list(np.zeros(len(impr), dtype=np.int)),
                        'search_for_item_session_ref_this_impr': list(np.zeros(len(impr), dtype=np.int)), 'clickout_item_session_ref_not_in_impr': 0,
                        'interaction_item_deals_session_ref_not_in_impr': 0, 'interaction_item_image_session_ref_not_in_impr': 0,
                        'interaction_item_info_session_ref_not_in_impr': 0, 'interaction_item_rating_session_ref_not_in_impr': 0,
                        'search_for_item_session_ref_not_in_impr': 0, 'session_length_in_step': 0,
                        'device': '', 'filters_when_clickout': '', 'session_length_in_time': 0, 'sort_order_active_when_clickout': 'sorted by default'}

            features['session_length_in_step'] = int(clk.step.values[0])
            features['device'] = clk['device'].values[0]

            if isinstance(clk.current_filters.values[0], str):
                features['filters_when_clickout'] = '|'.join(
                    [x + ' filter active when clickout' for x in clk.current_filters.values[0].split('|')])

            if len(x) > 1:
                features['session_length_in_time'] = abs(
                    int(clk['timestamp'].values[0]) - int(x.head(1)['timestamp'].values[0]))

                features['timing_last_action_before_clk'] = int(x.tail().timestamp.values[1]) - int(x.tail().timestamp.values[0])
            else:
                features['session_length_in_time'] = -1
                features['timing_last_action_before_clk'] = -1

            features['average_price_position'], \
            features['avg_price_interacted_item'], \
            features['avg_pos_interacted_items_in_impressions'], \
            features['pos_last_interaction_in_impressions'] = get_price_info_and_interaction_position(x, y)

            features['time_per_step'], features['frenzy_factor'] = get_frenzy_and_avg_time_per_step(x)

            times_doubleclickout_on_item = 0
            for item in set(y.reference.values):
                if len(y[y.reference == item]) > 1:
                    times_doubleclickout_on_item += 1

            features['times_doubleclickout_on_item'] = times_doubleclickout_on_item

            poi_search_df = x[x.action_type == 'search for poi']
            if poi_search_df.shape[0] > 0:
                last_poi_search_step = int(poi_search_df.tail(1).step.values[0])
                features['search_for_poi_distance_from_last_clickout'] = int(clk.step.values[0]) - last_poi_search_step
                features['search_for_poi_distance_from_first_action'] = last_poi_search_step - int(x.head(1).step.values[0])
            else:
                features['search_for_poi_distance_from_last_clickout'] = -1
                features['search_for_poi_distance_from_first_action'] = -1

            sort_change_df = x[x.action_type == 'change of sort order']
            if sort_change_df.shape[0] > 0:
                sort_change_step = int(sort_change_df.tail(1).step.values[0])
                features['change_sort_order_distance_from_last_clickout'] = int(clk.step.values[0]) - sort_change_step
                features['change_sort_order_distance_from_first_action'] = sort_change_step - int(
                    x.head(1).step.values[0])
            else:
                features['change_sort_order_distance_from_last_clickout'] = -1
                features['change_sort_order_distance_from_first_action'] = -1

            impr = clk['impressions'].values[0].split('|')
            prices = list(map(int, clk['prices'].values[0].split('|')))
            sorted_prices = prices.copy()
            sorted_prices.sort()

            change_of_sort_order_actions = x[x['action_type']
                                             == 'change of sort order']
            if len(change_of_sort_order_actions) > 0:
                change_of_sort_order_actions = change_of_sort_order_actions.tail(
                    1)
                features['sort_order_active_when_clickout'] = 'sort by ' + \
                    change_of_sort_order_actions['reference'].values[0]

            references = x['reference'].values
            actions = x['action_type'].values

            ## FREQUENCY EDIT - in case of presence of 'frequence' column in dataset
            if 'frequence' in x.columns.values:
                frequency = x['frequence'].values
            else:
                frequency = [1]*len(x)

            position_of_last_refence_on_impressions = None
            position_of_second_last_refence_on_impressions = None

            num_references = x[pd.to_numeric(x['reference'], errors='coerce').notnull()].drop_duplicates(subset = ['reference'], keep = 'last')

            if num_references.shape[0] > 0:
                last_num_reference = num_references.tail(1).reference.values[0]
                if last_num_reference in impr:
                    position_of_last_refence_on_impressions = impr.index(last_num_reference) + 1

            if num_references.shape[0] > 1:
                second_last_num_reference = num_references.tail(2).reference.values[1]
                if second_last_num_reference in impr:
                    position_of_second_last_refence_on_impressions = impr.index(second_last_num_reference) + 1

            not_to_cons_indices = []
            count = 0
            # Start features impressions
            for i in impr:
                indices = np.where(references == str(i))[0]

                not_to_cons_indices += list(indices)
                features['impression_position'].append(count+1)

                # Feature position wrt last interaction: if not exists a numeric reference in impressions,
                # default value is -999 (can't be -1 because values range is -24 to 24)
                if position_of_last_refence_on_impressions is not None:
                    features['impression_position_wrt_last_interaction'].append(count+1 - position_of_last_refence_on_impressions)
                else:
                    features['impression_position_wrt_last_interaction'].append(-999)

                if position_of_second_last_refence_on_impressions is not None:
                    features['impression_position_wrt_second_last_interaction'].append(count+1 - position_of_second_last_refence_on_impressions)
                else:
                    features['impression_position_wrt_second_last_interaction'].append(-999)

                features['price'].append(prices[count])
                features['price_position'].append(
                    sorted_prices.index(prices[count]))
                features['item_id'].append(int(i))

                cc = 0
                for jj in indices:
                    cc += int(frequency[jj])

                features['times_impression_appeared'].append(cc)

                if len(indices) > 0:
                    row_reference = x.head(indices[-1]+1).tail(1)
                    features['steps_from_last_time_impression_appeared'].append(
                        len(x)-indices[-1])
                    features['time_elapsed_from_last_time_impression_appeared'].append(
                        int(clk['timestamp'].values[0] - row_reference['timestamp'].values[0]))
                    features['kind_action_reference_appeared_last_time'].append(
                        'last_time_impression_appeared_as_' + row_reference['action_type'].values[0].replace(' ', '_'))
                    for idx in indices:
                        row_reference = x.head(idx+1).tail(1)

                        if 'frequence' in row_reference.columns.values:
                            freq = int(row_reference.frequence.values[0])
                        else:
                            freq = 1

                        features['_'.join(row_reference.action_type.values[0].split(
                            ' ')) + '_session_ref_this_impr'][count] += freq

                else:
                    features['steps_from_last_time_impression_appeared'].append(
                        0)
                    features['time_elapsed_from_last_time_impression_appeared'].append(
                        -1)
                    features['kind_action_reference_appeared_last_time'].append(
                        'last_time_reference_did_not_appeared')

                popularity = 0
                if int(i) in popularity_df:
                    popularity = popularity_df[int(i)]
                if clk['reference'].values[0] == i:
                    features['label'].append(1)
                    features['popularity'].append(popularity - 1)
                else:
                    features['label'].append(0)
                    features['popularity'].append(popularity)

                count += 1

            to_cons_indices = list(
                set(list(range(len(actions)))) - set(not_to_cons_indices))
            for ind in to_cons_indices:
                if (actions[ind].replace(' ', '_') + '_session_ref_not_in_impr') in features:
                    features[actions[ind].replace(
                        ' ', '_') + '_session_ref_not_in_impr'] += int(frequency[ind])

            return pd.DataFrame(features)

    def construct_features(df, q):
        dataset = df.groupby(['user_id', 'session_id']).progress_apply(func)

        one_hot = dataset['filters_when_clickout'].astype(
            str).str.get_dummies()
        missing = poss_filters - set(one_hot.columns)
        to_drop = set(one_hot.columns) - poss_filters

        for e in missing:
            one_hot[e] = 0
        for e in to_drop:
            one_hot = one_hot.drop([e], axis=1)
        dataset = dataset.drop(['filters_when_clickout'], axis=1)
        dataset = dataset.join(one_hot)

        # if the algorithm is xgboost, get the onehot of all the features. otherwise leave it categorical
        if algo == 'xgboost':
            one_hot = pd.get_dummies(dataset['device'])
            missing = poss_devices - set(one_hot.columns)
            for e in missing:
                one_hot[e] = 0
            dataset = dataset.drop(['device'], axis=1)
            dataset = dataset.join(one_hot)

            one_hot = pd.get_dummies(
                dataset['kind_action_reference_appeared_last_time'])
            missing = poss_actions - set(one_hot.columns)
            for e in missing:
                one_hot[e] = 0
            dataset = dataset.drop(
                ['kind_action_reference_appeared_last_time'], axis=1)
            dataset = dataset.join(one_hot)

            one_hot = pd.get_dummies(
                dataset['sort_order_active_when_clickout'])
            missing = poss_sort_orders - set(one_hot.columns)
            for e in missing:
                one_hot[e] = 0
            dataset = dataset.drop(
                ['sort_order_active_when_clickout'], axis=1)
            dataset = dataset.join(one_hot)


        dataset = dataset.reset_index().drop(['level_2'], axis=1)
        dataset = pd.merge(dataset, one_hot_accomodation, on=['item_id'])


        if 'item_id' in dataset.columns.values:
            dataset = dataset.drop(['item_id'], axis=1)
        if 'Unnamed: 0' in dataset.columns.values:
            dataset = dataset.drop(['Unnamed: 0'], axis=1)

        q.put(dataset)

    def save_features(features, count_chunk, target_session_id, target_user_id):
        print('started saving chunk {}'.format(count_chunk))
        test = features[features['user_id'].isin(
            target_user_id) & features['session_id'].isin(target_session_id)]
        train = features[(features['user_id'].isin(
            target_user_id) & features['session_id'].isin(target_session_id)) == False]

        if count_chunk == 1:
            path = 'dataset/preprocessed/{}/{}/{}/classification_train.csv'.format(cluster, mode, algo)
            check_folder(path)
            train.to_csv(path)

            path = 'dataset/preprocessed/{}/{}/{}/classification_test.csv'.format(cluster, mode, algo)
            check_folder(path)
            test.to_csv(path)
        else:
            with open('dataset/preprocessed/{}/{}/{}/classification_train.csv'.format(cluster, mode, algo), 'a') as f:
                 train.to_csv(f, header=False)
            with open('dataset/preprocessed/{}/{}/{}/classification_test.csv'.format(cluster, mode, algo), 'a') as f:
                test.to_csv(f, header=False)

        print('chunk {} over {} completed'.format(count_chunk, len(groups)))

    train = data.train_df(mode=mode, cluster=cluster)
    test = data.test_df(mode=mode, cluster=cluster)
    target_indices = data.target_indices(mode=mode, cluster=cluster)
    target_user_id = test.loc[target_indices]['user_id'].values
    target_session_id = test.loc[target_indices]['session_id'].values

    full = pd.concat([train, test])
    del train
    del test

    accomodations_df = data.accomodations_df()
    one_hot_accomodation = one_hot_of_accomodation(accomodations_df)

    #Popularity build on given full to avoid considering also clicks removed for test
    popularity_df = build_popularity(full)

    poss_filters = {'Free WiFi (Rooms) filter active when clickout', 'Casa Rural (ES) filter active when clickout',
                    'Singles filter active when clickout', 'Satisfactory Rating filter active when clickout',
                    'Laundry Service filter active when clickout', 'Bowling filter active when clickout',
                    'Deals + Beach (DE) filter active when clickout', 'Flatscreen TV filter active when clickout',
                    'Large Groups filter active when clickout', 'Safe (Hotel) filter active when clickout',
                    'Steam Room filter active when clickout', 'On-Site Boutique Shopping filter active when clickout',
                    'Hydrotherapy filter active when clickout', 'Fan filter active when clickout',
                    'Hammam filter active when clickout', 'All Inclusive (Upon Inquiry) filter active when clickout',
                    'Senior Travellers filter active when clickout', 'Shooting Sports filter active when clickout',
                    'Hostal (ES) filter active when clickout',
                    'Express Check-In / Check-Out filter active when clickout',
                    'Computer with Internet filter active when clickout', 'WiFi (Rooms) filter active when clickout',
                    'Openable Windows filter active when clickout', 'Good Rating filter active when clickout',
                    'Golf Course filter active when clickout', 'Self Catering filter active when clickout',
                    'Last Minute filter active when clickout', 'Sitting Area (Rooms) filter active when clickout',
                    'Hot Stone Massage filter active when clickout', 'Accessible Parking filter active when clickout',
                    'Bathtub filter active when clickout', 'Television filter active when clickout',
                    'Cable TV filter active when clickout', '5 Star filter active when clickout',
                    'Swimming Pool (Combined Filter) filter active when clickout',
                    'Small Hotel filter active when clickout', 'Deals + Beach (PT) filter active when clickout',
                    'Sort by Price filter active when clickout', 'Next Sunday filter active when clickout',
                    'Business Centre filter active when clickout', 'Pool Table filter active when clickout',
                    'Motel filter active when clickout', 'Microwave filter active when clickout',
                    'Washing Machine filter active when clickout', 'Towels filter active when clickout',
                    'Air Conditioning filter active when clickout', 'Deals + Beach (NL/BE) filter active when clickout',
                    'Breakfast Included filter active when clickout', 'Lift filter active when clickout',
                    'Ironing Board filter active when clickout', 'Mid-Size Hotel filter active when clickout',
                    'Hotel Chain filter active when clickout', 'Excellent Rating filter active when clickout',
                    'Tomorrow filter active when clickout', 'Adults Only filter active when clickout',
                    'Family Friendly filter active when clickout', '5 Nights filter active when clickout',
                    'Sauna filter active when clickout', 'Water Slide filter active when clickout',
                    'Hairdryer filter active when clickout', 'Beauty Salon filter active when clickout',
                    'Cheap filter active when clickout', 'Bungalows filter active when clickout',
                    'Top Deals filter active when clickout', 'This Weekend filter active when clickout',
                    'Wheelchair Accessible filter active when clickout', 'Body Treatments filter active when clickout',
                    'Non-Smoking Rooms filter active when clickout', 'Hairdresser filter active when clickout',
                    'Hotel filter active when clickout', 'Fitness filter active when clickout',
                    'Balcony filter active when clickout', 'Kitchen filter active when clickout',
                    'Sort By Distance filter active when clickout', 'Convenience Store filter active when clickout',
                    'Organised Activities filter active when clickout', 'Next Weekend filter active when clickout',
                    '4 Star filter active when clickout', 'Szep Kartya filter active when clickout',
                    'Halal Food filter active when clickout', 'Club Hotel filter active when clickout',
                    'Deals + Beach (IT) filter active when clickout', 'Safe (Rooms) filter active when clickout',
                    'Reception (24/7) filter active when clickout', 'Car Park filter active when clickout',
                    'Conference Rooms filter active when clickout', 'From 2 Stars filter active when clickout',
                    'Sun Umbrellas filter active when clickout', 'Next Monday filter active when clickout',
                    'Airport Hotel filter active when clickout', 'Bed & Breakfast filter active when clickout',
                    'Desk filter active when clickout', 'Today filter active when clickout',
                    'From 3 Stars filter active when clickout', 'Deals + Beach (TR) filter active when clickout',
                    'Hiking Trail filter active when clickout', 'Hypoallergenic Bedding filter active when clickout',
                    'Design Hotel filter active when clickout', 'Beach filter active when clickout',
                    'Central Heating filter active when clickout', 'Kosher Food filter active when clickout',
                    'Tennis Court (Indoor) filter active when clickout', 'Boat Rental filter active when clickout',
                    'Deals + Beach (DK) filter active when clickout', 'Pay-TV filter active when clickout',
                    'Shower filter active when clickout', 'WiFi (Public Areas) filter active when clickout',
                    "Kids' Club filter active when clickout", 'Theme Hotel filter active when clickout',
                    'Sort By Popularity filter active when clickout', 'Camping Site filter active when clickout',
                    'Teleprinter filter active when clickout', 'OFF - Rating Good filter active when clickout',
                    'Doctor On-Site filter active when clickout', 'From 4 Stars filter active when clickout',
                    'Guest House filter active when clickout', 'Best Value filter active when clickout',
                    'Swimming Pool (Indoor) filter active when clickout', 'Onsen filter active when clickout',
                    'Volleyball filter active when clickout', '3 Star filter active when clickout',
                    'Table Tennis filter active when clickout', 'Deals + Beach (AR) filter active when clickout',
                    'Pet Friendly filter active when clickout', 'Swimming Pool (Bar) filter active when clickout',
                    'Hotel Bar filter active when clickout', 'Pousada (BR) filter active when clickout',
                    'Room Service (24/7) filter active when clickout', 'Concierge filter active when clickout',
                    'Direct beach access filter active when clickout', 'Gay Friendly filter active when clickout',
                    'Playground filter active when clickout', 'Luxury Hotel filter active when clickout',
                    'Deck Chairs filter active when clickout', 'Sailing filter active when clickout',
                    'Room Service filter active when clickout', 'Massage filter active when clickout',
                    'Next Friday filter active when clickout', 'Skiing filter active when clickout',
                    'Internet (Rooms) filter active when clickout', 'Next Saturday filter active when clickout',
                    'Romantic filter active when clickout', 'Surfing filter active when clickout',
                    '3 Nights filter active when clickout', 'Hostel filter active when clickout',
                    'Resort filter active when clickout', 'Terrace (Hotel) filter active when clickout',
                    'Disneyland Paris filter active when clickout', 'Bike Rental filter active when clickout',
                    'Jacuzzi (Hotel) filter active when clickout', 'Free WiFi (Combined) filter active when clickout',
                    'Solarium filter active when clickout', 'Diving filter active when clickout',
                    'Free WiFi (Public Areas) filter active when clickout', 'Best Rates filter active when clickout',
                    'Health Retreat filter active when clickout', 'Tennis Court filter active when clickout',
                    'Hypoallergenic Rooms filter active when clickout', 'Very Good Rating filter active when clickout',
                    'Large Hotel filter active when clickout', 'Airport Shuttle filter active when clickout',
                    'Casino (Hotel) filter active when clickout', 'Farmstay filter active when clickout',
                    'Horse Riding filter active when clickout', 'Country Hotel filter active when clickout',
                    'House / Apartment filter active when clickout', 'Disneyland filter active when clickout',
                    'Focus on Distance filter active when clickout',
                    'Swimming Pool (Outdoor) filter active when clickout', 'Holiday filter active when clickout',
                    'Deals + Beach (MX) filter active when clickout', '1 Star filter active when clickout',
                    'Radio filter active when clickout', 'Telephone filter active when clickout',
                    'Focus on Rating filter active when clickout', 'This Monday filter active when clickout',
                    '1 Night filter active when clickout', 'Satellite TV filter active when clickout',
                    'Porter filter active when clickout', '2 Nights filter active when clickout',
                    '2 Star filter active when clickout', 'Cot filter active when clickout',
                    'Gym filter active when clickout', 'Convention Hotel filter active when clickout',
                    'Restaurant filter active when clickout', 'Fridge filter active when clickout',
                    'Serviced Apartment filter active when clickout', 'Minigolf filter active when clickout',
                    'Eco-Friendly hotel filter active when clickout', 'Nightclub filter active when clickout',
                    'Cosmetic Mirror filter active when clickout', 'Ski Resort filter active when clickout',
                    'Deals + Beach (JP) filter active when clickout', 'Childcare filter active when clickout',
                    'Sort By Rating filter active when clickout', 'Boutique Hotel filter active when clickout',
                    'Accessible Hotel filter active when clickout', 'Beach Bar filter active when clickout',
                    'Electric Kettle filter active when clickout', 'OFF - Rating Very Good filter active when clickout',
                    'Deals + Beach (GR) filter active when clickout',
                    'Spa (Wellness Facility) filter active when clickout', 'Honeymoon filter active when clickout',
                    'Business Hotel filter active when clickout', 'Spa Hotel filter active when clickout'}
    poss_devices = {'mobile', 'desktop', 'tablet'}
    poss_actions = {'last_time_reference_did_not_appeared', 'last_time_impression_appeared_as_clickout_item',
                    'last_time_impression_appeared_as_interaction_item_deals',
                    'last_time_impression_appeared_as_interaction_item_image',
                    'last_time_impression_appeared_as_interaction_item_info',
                    'last_time_impression_appeared_as_interaction_item_rating',
                    'last_time_impression_appeared_as_search_for_item'}

    poss_sort_orders = {'sort by distance and recommended', 'sort by our recommendations',
                        'sort by rating and recommended', 'sort by rating only',
                        'sort by price and recommended', 'sort by price only',
                        'sort by distance only', 'sort by interaction sort button'}

    # build in chunk
    count_chunk = 0
    chunk_size = 22000

    groups = full.groupby(np.arange(len(full))//chunk_size)
    # create the dataset in parallel threads: one thread saves, the other create the dataset for the actual group 
    p2 = None
    for idxs, gr in groups:
        q = Queue()
        p1 = Process(target=construct_features, args=(gr,q,))
        p1.start()
        if p2 != None:
            p2.join()
        features = q.get()
        p1.join()
        count_chunk += 1
        p2 = Process(target=save_features, args=(features, count_chunk, target_session_id, target_user_id,))
        p2.start()

if __name__ == "__main__":
    build_dataset(mode='small', cluster='no_cluster', algo='xgboost')
