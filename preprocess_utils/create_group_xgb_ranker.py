import numpy as np
import data

def create_groups(df, mode, cluster):
    train = df.sort_values(['user_id', 'session_id'])
    count = 1
    user_id = ''
    for idx, row in train.iterrows():
        if user_id == '':
            user_id = row.user_id
        if user_id != row.user_id:
            group.append(count - 1)
            user_id = row.user_id
            count = 1
        count += 1
    group.append(count - 1)
    group = np.array(group)
    np.save('dataset/preprocessed/{}/{}/xgboost/group'.format(cluster, mode))

if __name__ == "__main__":
    mode = 'small'
    cluster = 'no_cluster'
    df = data.classification_train_df(mode=mode, sparse=True, cluster=cluster)
    create_groups(df, mode, cluster)
