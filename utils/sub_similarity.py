import pandas as pd
import numpy as np

def compute_sub_similarity(sub1, sub2):
    def _compute_sub_similarity(sub1, sub2, weights=None):
        df = pd.merge(sub1, sub2, on=['user_id', 'session_id', 'timestamp', 'step'])
        x = np.array(
            list(map(lambda arr: np.array(list(map(int, arr.split(' ')))), df['item_recommendations_x'].values)))
        y = np.array(
            list(map(lambda arr: np.array(list(map(int, arr.split(' ')))), df['item_recommendations_y'].values)))
        arr_diff = x - y
        pad = np.array(list(map(lambda arr: np.append(arr, np.zeros(25 - len(arr))), arr_diff)))
        no_w = np.clip(np.absolute(pad), a_min=0, a_max=1)
        if weights is None:
            weights = 1 / np.arange(1, 26, 1)
        else:
            weights = weights
        arr_weighted = no_w * weights
        normalized = arr_weighted / np.sum(weights)
        res = np.sum(normalized) / arr_weighted.shape[0]
        print(f'   distance is {res}')

    _compute_sub_similarity(sub1, sub2)
    print('lets see how much are different the various columns...')
    w = np.zeros(25)
    for i in range(10):
        print(f'perc of different elem on col {i}')
        w[i:i + 1] = 1
        _compute_sub_similarity(sub1, sub2, weights=w)

if __name__ == '__main__':
    pass