from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
import data
import pandas as pd
import numpy as np
from tqdm import tqdm


class MaxPositionInteractedReference(FeatureBase):
    """
    max position on the impressions list of the reference with which the user interacted with
    | user_id | session_id | max_position_interacted_reference
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'max_position_interacted_reference'
        super(MaxPositionInteractedReference, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def RepresentsInt(self, s):
        try: 
            int(s)
            return True
        except ValueError:
            return False

    def extract_feature(self):
        tr = data.train_df('small')
        te = data.test_df('small')
        df = pd.concat([tr, te])
        clk_indices = sorted(find(df))
        last_pos_interacted = []
        for idx in tqdm(clk_indices):
            impressions = df.at[idx, 'impressions']
            impressions = list(map(int, impressions.split('|')))
            a_user = df.at[idx, 'user_id']
            a_sess = df.at[idx, 'session_id']
            last_pos = -2
            j = idx-1
            while j >= 0:
                try:
                    new_user = df.at[j, 'user_id']
                    new_sess = df.at[j, 'session_id']
                    if new_user == a_user and new_sess == a_sess:
                        reference = df.at[j, 'reference']
                        if self.RepresentsInt(reference):
                            candidate_last_pos = impressions.index(int(reference))
                            if candidate_last_pos > last_pos:
                                last_pos = candidate_last_pos
                    else:
                        break
                    j -= 1
                except:
                    j -= 1
            last_pos_interacted.append(last_pos+1)

        final = df.loc[clk_indices][['user_id', 'session_id']]
        final['max_position_interacted_references'] = last_pos_interacted
        return final.reset_index(drop=True)

if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = MaxPositionInteractedReference(mode=mode, cluster='no_cluster')
    c.save_feature()
