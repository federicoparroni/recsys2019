import numpy as np
import data
from tqdm import tqdm

def groups(users):
    au = users[0]
    groups = []
    count = 1
    for u in tqdm(users):
        if u != au:
            groups.append(count - 1)
            count = 1
            au = u
        count += 1
    groups.append(count - 1)
    return groups

def create_groups(df, mode, cluster):
    users = df['user_id'].to_dense().values
    print('data are ready')
    group = groups(list(users))
    np.save('dataset/preprocessed/{}/{}/xgboost/group'.format(cluster, mode), group)

if __name__ == "__main__":
    mode = 'small'
    cluster = 'no_cluster'
    df = data.classification_train_df(mode=mode, sparse=True, cluster=cluster)
    create_groups(df, mode, cluster)
