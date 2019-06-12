import pandas as pd
import data
from clusterize.clusterize_base import ClusterizeBase
from tqdm.auto import tqdm
tqdm.pandas()
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.retrieve_real_test_indices import retrieve_real_test_indices

class EvalOnNoNumericReferenceNoOneStep(ClusterizeBase):

    def __init__(self):
        super(EvalOnNoNumericReferenceNoOneStep, self).__init__()
        self.name = 'eval_on_no_numeric_reference_no_one_step'

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

        train = data.train_df('small')

        test = data.test_df('small')
        tgt_indices = data.target_indices('small')

        real_test_to_keep = []
        for idx in tgt_indices:
            usr_sess_indices = []
            theres_int = False
            a_user = test.at[idx, 'user_id']
            a_sess = test.at[idx, 'session_id']
            usr_sess_indices.append(idx)
            j = idx-1
            pos_moved = 0
            while j >= 0:
                try:
                    new_user = test.at[j, 'user_id']
                    new_sess = test.at[j, 'session_id']
                    if new_user == a_user and new_sess == a_sess:
                        usr_sess_indices.append(j)
                        reference = test.at[j, 'reference']
                        if RepresentsInt(reference):
                            theres_int = True
                        j -= 1
                        pos_moved += 1
                    else:
                        if not (pos_moved == 0 or theres_int):
                            real_test_to_keep += sorted(usr_sess_indices)
                        break
                except:
                    if j < test.index.values[0]:
                        if not (pos_moved == 0 or theres_int):
                            real_test_to_keep += sorted(usr_sess_indices)
                        break
                    else:
                        j -= 1
        
        self.train_indices = train.index.values
        real_test_indices = retrieve_real_test_indices(mode, 'no_cluster')
        all_test_indices = data.test_df(mode).index.values
        self.test_indices = sorted(list(set(all_test_indices) - set(real_test_indices)) + real_test_to_keep)
        self.target_indices = sorted(list(set(self.test_indices) & set(tgt_indices)))


if __name__ == '__main__':
    import utils.menu as menu

    obj = EvalOnNoNumericReferenceNoOneStep()

    mode = menu.mode_selection()

    obj.save(mode, add_unused_clickouts_to_test=False)
