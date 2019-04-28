import numpy as np
import data
from tqdm import tqdm

def create_groups(df, mode, cluster):
    train = df[['user_id', 'session_id', 'impression_position']].to_dense()
    train = train.sort_values(['user_id', 'session_id'])
    group = train.groupby(['user_id']).apply(lambda x: len(x)).values
    np.save('dataset/preprocessed/{}/{}/xgboost/group'.format(cluster, mode), group)

if __name__ == "__main__":
    mode = 'small'
    cluster = 'no_cluster'
    df = data.classification_train_df(mode=mode, sparse=True, cluster=cluster)
    create_groups(df, mode, cluster)
