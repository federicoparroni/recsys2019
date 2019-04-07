from recommenders.recommender_base import RecommenderBase
import numpy as np
import pandas as pd
import data
import xgboost as xgb
from tqdm import tqdm


class XGBoost(RecommenderBase):

    def __init__(self, mode, cluster='no_cluster'):
        tqdm.pandas()

        name = 'xgboost'
        super(XGBoost, self).__init__(name=name, mode=mode, cluster=cluster)

        self.preds = None
        self.target_indices = data.target_indices(mode=mode, cluster=cluster)

        df_train = data.train_df('small')
        self.train_len = len(df_train)
        print(f'len df train: {len(df_train)}')
        df_test = data.test_df('small')
        print(f'len df test: {len(df_test)}')
        df = pd.concat([df_train, df_test])
        print(f'len df : {len(df)}')

        dataset = df.groupby(['user_id', 'session_id']).progress_apply(self._extract_features)
        one_hot = pd.get_dummies(dataset['kind_action_reference_appeared'])
        dataset = dataset.drop(['kind_action_reference_appeared'], axis=1)
        self.dataset = dataset.join(one_hot)


    def fit(self):
        X, y = self.dataset.iloc[:, [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]], self.dataset.iloc[:, 4]

        # split the data
        X_train = X.iloc[:791373, :]
        X_test = X.iloc[791373:, :]
        y_train = y.iloc[:791373]
        #y_test = y.iloc[791373:]

        xg_reg = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100)
        xg_reg.fit(X_train, y_train)
        preds = xg_reg.predict_proba(X_test)

        self.preds = [a[1] for a in preds]


    def _extract_features(self, x, type = 'train'):
        y = x[x['action_type'] == 'clickout item']
        if len(y) > 0:
            clk = y.tail(1)
            head_index = x.head(1).index

            # considering only the past!
            x = x.loc[head_index.values[0]:clk.index.values[0] - 1]

            impr = clk['impressions'].values[0].split('|')
            prices = list(map(int, clk['prices'].values[0].split('|')))
            sorted_prices = prices.copy()
            sorted_prices.sort()

            references = x['reference'].values

            # features
            features = {'times_impression_appeared': [],
                        'time_elapsed_from_last_time_impression_appeared': [],
                        'steps_from_last_time_impression_appeared': [],
                        'kind_action_reference_appeared': [], 'impression_position': [], 'label': [], 'price': [],
                        'price_position': []}
            count = 0
            for i in impr:
                indices = np.where(references == str(i))[0]

                features['impression_position'].append(count + 1)
                features['price'].append(prices[count])
                features['price_position'].append(sorted_prices.index(prices[count]))
                if len(indices) > 0:
                    row_reference = x.head(indices[-1] + 1).tail(1)
                    features['steps_from_last_time_impression_appeared'].append(len(x) - indices[-1])
                    features['time_elapsed_from_last_time_impression_appeared'].append(
                        int(clk['timestamp'].values[0] - row_reference['timestamp'].values[0]))
                    features['kind_action_reference_appeared'].append(row_reference['action_type'].values[0])
                else:
                    features['steps_from_last_time_impression_appeared'].append(0)
                    features['time_elapsed_from_last_time_impression_appeared'].append(np.inf)
                    features['kind_action_reference_appeared'].append('no_action')
                features['times_impression_appeared'].append(len(indices))

                if type == 'train':
                    if clk['reference'].values[0] == i:
                        features['label'].append(1)
                    else:
                        features['label'].append(0)
                count += 1

            return pd.DataFrame(features)


    def get_scores_batch(self):
        pass

    def recommend_batch(self):
        current_index = 0
        print('loading full df...')
        full_df = data.full_df()
        print('done')

        predictions = []
        for index in tqdm(self.target_indices):
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            scores = self.preds[current_index:len(impr)]
            current_index = len(impr)

            sorted_impr = [impr for _, impr in sorted(zip(scores, impr))]
            predictions.append((index, sorted_impr))
        return predictions


if __name__ == '__main__':
    model = XGBoost(mode='small', cluster='no_cluster')
    model.evaluate()






