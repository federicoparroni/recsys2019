import pandas as pd
import data
from clusterize.clusterize_base import ClusterizeBase
from tqdm.auto import tqdm
tqdm.pandas()
from preprocess_utils.last_clickout_indices import find

class NoNumericReferenceNoOneStep(ClusterizeBase):

    def __init__(self):
        super(NoNumericReferenceNoOneStep, self).__init__()
        self.name = 'no_numeric_reference_no_one_step'

    def _fit(self, mode):
        """
        has target indices equal to the 10% of the session with:
        - no num ref
        - more than 1 step
        but anyway we train on all of them ;)
        """

        def RepresentsInt(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        train = data.train_df(mode)
        self.train_indices = train.index.values

        test = data.test_df(mode)
        test_index = test.index.values
        tgt_indices = data.target_indices(mode)

        real_test_to_drop = []
        for idx in tgt_indices:
            usr_sess_indices = []
            theres_int = False
            try:
                a_user = df.at[idx, 'user_id']
                a_sess = df.at[idx, 'session_id']
                usr_sess_indices.append(idx)
            except:
                continue
            j = idx-1
            while j >= 0:
                try:
                    new_user = df.at[j, 'user_id']
                    new_sess = df.at[j, 'session_id']
                    if new_user == a_user and new_sess == a_sess:
                        usr_sess_indices.append(j)
                        reference = df.at[j, 'reference']
                        if RepresentsInt(reference):
                            theres_int = True
                        j -= 1
                    else:
                        # add indices 
                        if not (idx-j >= 2 and not theres_int):
                            real_test_to_drop += usr_sess_indices
                        break
                except:
                    j -= 1
        
        self.train_indices = sorted(list(set(train_index) & set(to_return)))
        self.test_indices = sorted(list(set(test_index) & set(to_return)))
        self.target_indices =  sorted(list(set(tgt_indices) & set(to_return)))


if __name__ == '__main__':
    import utils.menu as menu

    obj = NoNumericReferenceNoOneStep()

    mode = menu.mode_selection()

    obj.save(mode, add_unused_clickouts_to_test=False)
