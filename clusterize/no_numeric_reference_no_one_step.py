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
        train, test and target indices are just sessions which have:
        - no num ref
        - more than 1 step
        """

        def RepresentsInt(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        train = data.train_df(mode)
        train_index = train.index.values
        test = data.test_df(mode)
        test_index = test.index.values
        tgt_indices = data.target_indices(mode)
        df = pd.concat([train, test])
        del train
        del test
        lst_clk_indices = sorted(find(df))

        to_return = []
        for idx in lst_clk_indices:
            usr_sess_indices = []
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
                            break
                        j -= 1
                    else:
                        if idx-j >= 2:
                            to_return += usr_sess_indices
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
