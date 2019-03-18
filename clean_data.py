import pandas as pd
import data
import numpy as np
from tqdm import tqdm
import utils.check_folder as cf
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sps
import utils.check_folder as cf

def merge_consecutive_equal_actions():
    tqdm.pandas()
    test = data.test_df('full')
    test_grouped_by_session_id = test.groupby('session_id')
    merged = test_grouped_by_session_id.progress_apply(_merge_consecutive_equal_actions)
    cf.check_folder('dataset/cleaned_csv')
    merged.to_csv('dataset/cleaned_csv/test.csv')

def _merge_consecutive_equal_actions(df):
    df_cleaned = pd.DataFrame(columns=['user_id', 'session_id', 'timestamp', 'step', 'action_type',
                                       'occurences', 'reference', 'platform', 'city', 'device', 'current_filters',
                                       'impressions', 'prices'])
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if i == 0:
            row['occurences'] = 1
            df_cleaned = df_cleaned.append(row, ignore_index=True)
        else:
            a_r = np.array(row[['action_type', 'reference']])
            last = np.array(df_cleaned.tail(1)[['action_type', 'reference']])
            if (a_r == last).all():
                df_cleaned.at[df_cleaned.tail(1).index[0], 'occurences'] += 1
                df_cleaned.tail(1)[['timestamp', 'step']] = row[['timestamp', 'step']].values
            else:
                row['occurences'] = 1
                df_cleaned = df_cleaned.append(row)
    return df_cleaned


attribute_dict = {}


def list_item_metadata():
    tqdm.pandas()
    data.accomodations_df()['properties'].progress_apply(_list_item)
    attribute_frequency_tuples = sorted(attribute_dict.items(), key=lambda x: x[1])

    attribute_df = pd.DataFrame(attribute_frequency_tuples, columns=['attribute', 'frequency'])

    #plot the attribute - frequency graphic
    plot_df(attribute_df, 3, 'bar', 'attribute')
    attribute_df.to_csv('attributes_apartament', index=False)


def _list_item(r):
    global attribute_dict
    if isinstance(r, str):
        temp = r.split('|')
        for attribute in temp:
            if attribute in attribute_dict.keys():
                attribute_dict[attribute] += 1
            else:
                attribute_dict[attribute] = 1


def plot_df(df, blocks_number, kind, x):
    block_dimension = math.ceil(df.shape[0]/blocks_number)
    for i in range(0, df.shape[0], block_dimension):
        if i+block_dimension < df.shape[0]:
            df[i:i+block_dimension].plot(kind=kind, x=x)
        else:
            df[i:df.shape[0]].plot(kind=kind, x=x)
        plt.show()


