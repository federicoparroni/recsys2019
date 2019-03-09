import pandas as pd
import out
import data


w = -1
z = 0

def rule_based_recommender(test_df, handle_df):
    global z
    test_df.drop(columns = ['platform', 'city', 'device', 'current_filters'], inplace=True)
    user_sessions = test_df.groupby('user_id')
    z = len(user_sessions)
    user_sessions = user_sessions.apply(_user_sessions_recommender)
    final_list = []
    for i in user_sessions.tolist():
        for j in i:
            final_list.append(j)
    out.create_sub(final_list, handle_df, eval=False)

def _user_sessions_recommender(df):
    global w
    w+=1
    print(w*100/z)
    df = df.reset_index()
    prediction_rows = df[(df['action_type'] == 'clickout item') & (df['reference'].isnull())]

    indeces = list(map(int, prediction_rows.index))

    f_pred = []
    # print(indeces)
    past_index = 0
    for i in range(len(indeces)):
        row = prediction_rows.iloc[i]
        sliced_df = df.iloc[past_index:indeces[i] + 1]
        past_index = indeces[i]+1
        impressions = list(map(int, row['impressions'].split('|')))
        prices = list(map(int, row['prices'].split('|')))
        session_id = sliced_df['session_id'].values[0]
        predictions = []
        for i in range(sliced_df.shape[0] - 1, -1, -1):
            r = sliced_df.iloc[i]
            try:
                ref = int(r['reference'])
                indx = impressions.index(ref)
            except ValueError:
                continue
            predictions.append(impressions.pop(indx))
            _ = prices.pop(indx)
        predictions += [x for _, x in sorted(zip(prices, impressions), key=lambda pair: pair[0])]
        f_pred.append((session_id, predictions))
    return f_pred

if __name__ == '__main__':
    test = data.test_df()
    handle = data.handle_df()
    #test = pd.read_csv('dataset/preprocessed/test_small.csv')
    #handle = pd.read_csv('dataset/preprocessed/handle_small.csv')
    rule_based_recommender(test, handle)