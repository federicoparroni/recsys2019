import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

"""
creates train and test dataframe that can be used for classification,
starting from train_df and test_df of the specified cluster and mode

the resulting dataframe have structure:
user_id | session_id | accomodation_id | feature_1 | ....  | feature_n | label

the accomodation_ids are the ones showing up in the impressions
label is 1 in case the accomodation is the one clicked in the clickout
"""
def build_dataset(mode, cluster='no_cluster'):
    
    # build the onehot of accomodations attributes
    def one_hot_of_accomodation(accomodations_df):
        accomodations_df.properties = accomodations_df.properties.progress_apply(
            lambda x: x.split('|') if isinstance(x, str) else x)
        accomodations_df.fillna(value='', inplace=True)
        mlb = MultiLabelBinarizer()
        one_hot_features = mlb.fit_transform(accomodations_df.properties)
        one_hot_accomodations_df = pd.DataFrame(one_hot_features, columns=mlb.classes_)
        attributes_df = pd.concat([accomodations_df.drop('properties', axis=1), one_hot_accomodations_df], axis=1)
        return attributes_df

    def build_popularity(accomodations_df, df):
        df = df[(df['action_type'] == 'clickout item') & (~df['reference'].isnull())]
        clicked_references = list(map(int, list(df['reference'].values)))
        popularity_df = accomodations_df.drop(['properties'], axis=1).set_index('item_id')
        popularity_df['popularity_impression'] = 0
        for e in tqdm(clicked_references):
            popularity_df.loc[e]['popularity_impression'] += 1
        return popularity_df

    def func(x):
        # TO-DO: user features! ;)
        y = x[(x['action_type'] == 'clickout item')]
        if len(y) > 0:
            clk = y.tail(1)
            head_index = x.head(1).index

            # features
            features = {'label': [], 'times_impression_appeared': [], 'time_elapsed_from_last_time_impression_appeared': [], 'impression_position': [],
                        'steps_from_last_time_impression_appeared': [], 'kind_action_reference_appeared_last_time': [], 
                        'price': [], 'price_position': [],
                        'item_id': [], 'popularity': [], 'other change of sort order session': 0, 'other clickout item session': 0, 
                        'other filter selection session': 0, 'other interaction item deals session': 0, 'other interaction item image session': 0, 
                        'other interaction item info session': 0, 'other interaction item rating session': 0, 'other search for destination session': 0, 
                        'other search for item session': 0, 'other search for poi session': 0, 'session length': 0,
                        'device': '', 'filters_when_clickout': '', 'session_length': 0}
            
            features['session length'] = len(x)
            features['device'] = clk['device'].values[0]
            features['filters_when_clickout'] = clk['current_filters'].values[0]
            features['session_length'] = abs(int(clk['timestamp'].values[0]) - int(x.head(1)['timestamp'].values[0]))
            
            # considering only the past!
            x = x.loc[head_index.values[0]:clk.index.values[0]-1]
            
            impr = clk['impressions'].values[0].split('|')
            prices = list(map(int, clk['prices'].values[0].split('|')))
            sorted_prices = prices.copy()
            sorted_prices.sort()
            
            references = x['reference'].values
            actions = x['action_type'].values
            not_to_cons_indices = []
            
            count = 0
            for i in impr:
                indices = np.where(references == str(i))[0]
                not_to_cons_indices += list(indices)
                features['impression_position'].append(count+1)
                features['price'].append(prices[count])
                features['price_position'].append(sorted_prices.index(prices[count]))
                features['item_id'].append(int(i))
                features['times_impression_appeared'].append(len(indices))
                if len(indices) > 0:
                    row_reference = x.head(indices[-1]+1).tail(1)
                    features['steps_from_last_time_impression_appeared'].append(len(x)-indices[-1])
                    features['time_elapsed_from_last_time_impression_appeared'].append(int(clk['timestamp'].values[0] - row_reference['timestamp'].values[0]))
                    features['kind_action_reference_appeared_last_time'].append('last_time_impression_appeared_as_' + row_reference['action_type'].values[0])
                else:
                    features['steps_from_last_time_impression_appeared'].append(0)
                    features['time_elapsed_from_last_time_impression_appeared'].append(-1)
                    features['kind_action_reference_appeared_last_time'].append('last_time_reference_did_not_appeared')
                
                if clk['reference'].values[0] == i:
                    features['label'].append(1)
                    features['popularity'].append(popularity_df.loc[int(i)].values[0] - 1)
                else:
                    features['label'].append(0)
                    features['popularity'].append(popularity_df.loc[int(i)].values[0])

                count += 1
            
            to_cons_indices = list(set(list(range(len(actions)))) - set(not_to_cons_indices))
            for ind in to_cons_indices:
                features['other ' + actions[ind] + ' session'] += 1

            return pd.DataFrame(features)

    def construct_features(df):
        dataset = df.groupby(['user_id', 'session_id']).progress_apply(func)
        dataset = dataset.to_sparse(fill_value=0)

        one_hot = pd.get_dummies(dataset['kind_action_reference_appeared_last_time'], sparse=True)
        dataset = dataset.drop(['kind_action_reference_appeared_last_time'],axis = 1)
        dataset = dataset.join(one_hot)

        one_hot = pd.get_dummies(dataset['device'], sparse=True)
        dataset = dataset.drop(['device'], axis=1)
        dataset = dataset.join(one_hot)
        
        one_hot_dense = dataset['filters_when_clickout'].str.get_dummies()
        one_hot = one_hot_dense.to_sparse(fill_value=0)
        del one_hot_dense
        dataset = dataset.drop(['filters_when_clickout'],axis = 1)
        dataset = dataset.join(one_hot)

        dataset = dataset.reset_index().drop(['level_2'], axis=1)
        dataset = pd.merge(dataset, one_hot_accomodation, on=['item_id'])

        return dataset


    train = data.train_df(mode=mode, cluster=cluster)
    test = data.test_df(mode=mode, cluster=cluster)
    target_indices = data.target_indices(mode=mode, cluster=cluster)
    target_user_id = test.loc[target_indices]['user_id'].values
    target_session_id = test.loc[target_indices]['session_id'].values

    full = pd.concat([train, test])

    accomodations_df = data.accomodations_df()
    one_hot_accomodation_dense = one_hot_of_accomodation(accomodations_df)
    one_hot_accomodation = one_hot_accomodation_dense.to_sparse(fill_value=0)
    del one_hot_accomodation_dense

    popularity_df = build_popularity(accomodations_df, full)
    features_full = construct_features(full)

    test = features_full[features_full['user_id'].isin(target_user_id) & features_full['session_id'].isin(target_session_id)]
    train = features_full[(features_full['user_id'].isin(target_user_id) & features_full['session_id'].isin(target_session_id)) == False]

    train.to_csv('dataset/preprocessed/{}/{}/classification_train.csv'.format(cluster, mode))
    print('training classification dataset created in ' + 'dataset/preprocessed/{}/{}/classification_train.csv'.format(cluster, mode))
    test.to_csv('dataset/preprocessed/{}/{}/classification_test.csv'.format(cluster, mode))
    print('testing classification dataset created in ' + 'dataset/preprocessed/{}/{}/classification_test.csv'.format(cluster, mode))


if __name__ == "__main__":
    build_dataset(mode='small')
