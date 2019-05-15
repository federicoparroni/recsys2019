import sys
import os

sys.path.append(os.getcwd())

from clusterize.clusterize_base import ClusterizeBase
import data
import utils.menu as menu
from tqdm.auto import tqdm
import numpy


class ClusterNoNumRef(ClusterizeBase):
    """
    Cluster in which the sessions have no numeric reference
    """

    def __init__(self):
        super(ClusterNoNumRef, self).__init__('cluster_no_num_ref')

    def _fit(self, mode):
        tqdm.pandas()

        def find_indices_no_num_ref(df, mode):
            assert mode in ['train',
                            'test'], f'you have passed a wrong mode...\n it has to be: train or test\n recived:{mode}'

            # fill the nan values with an empty string
            df = df.fillna('')

            # TRAIN -> last clickout of the session is present
            # we need that in the whole session have to be exactly 1 num ref (the one of the click)
            # TEST -> last clickout of the session is NOT present
            # we need to have 0 num ref in the session
            if mode == 'train':
                _N_NUMERIC_REF_FOUND = 1
            else:
                _N_NUMERIC_REF_FOUND = 0

            indices = []
            temp_df = df[['user_id', 'session_id', 'reference', 'action_type']]
            temp_df = temp_df.sort_index()

            # set the current user and current session to the last found on df
            cur_ses = ''
            cur_user = ''

            # initialize the variable
            n_numeric_reference_found = 0
            clickout_found = False
            idx_list_temp = []

            for idx in tqdm(temp_df.index.values[::-1]):
                # retrieve the current user and session
                ruid = temp_df.at[idx, 'user_id']
                rsid = temp_df.at[idx, 'session_id']

                # check if the session has changed
                if (ruid != cur_user or rsid != cur_ses):

                    # if the session is without numeric reference but has a clickout
                    if (n_numeric_reference_found == _N_NUMERIC_REF_FOUND) and clickout_found and len(idx_list_temp)>1:
                        indices = indices + idx_list_temp

                    # reset the value of the variables
                    n_numeric_reference_found = 0
                    clickout_found = False
                    idx_list_temp = []

                    # update the current user and session
                    cur_user = ruid
                    cur_ses = rsid

                rref = temp_df.at[idx, 'reference'].isdigit() * 1
                ract = temp_df.at[idx, 'action_type'] == 'clickout item'

                n_numeric_reference_found += rref
                clickout_found = clickout_found or ract

                # remove all the interactions after the clickout
                if clickout_found:
                    idx_list_temp.append(idx)

            return indices[::-1]

        train_df = data.train_df(mode)
        self.train_indices = find_indices_no_num_ref(train_df, 'train')
        del train_df

        test_df = data.test_df(mode)
        self.test_indices = find_indices_no_num_ref(test_df, 'test')

        self.target_indices = []


if __name__ == '__main__':
    obj = ClusterNoNumRef()

    mode = menu.mode_selection()
    obj.save(mode)
