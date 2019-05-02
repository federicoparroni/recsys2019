import pandas as pd
import data
from clusterize.clusterize_base import ClusterizeBase
from tqdm.auto import tqdm

tqdm.pandas()

import os
os.chdir("recsys2019/")
print(os.getcwd())

class ClusterSessionsWithoutNumericalReferences(ClusterizeBase):

    def __init__(self):
        super(ClusterSessionsWithoutNumericalReferences, self).__init__()
        self.name = 'cluster_sessions_no_numerical_reference'

    def func_remove_steps_over_clk(self, x):
        y = x[x['action_type'] == 'clickout item']

        clk = y.tail(1)

        if len(y) == 1 and (x[pd.to_numeric(x['reference'], errors='coerce').notnull()].shape[0] == 1):
            # keep only sessions until last clickout
            x = x[x["step"] <= int(clk["step"])]
            return pd.DataFrame(x)

    def func_remove_steps_over_clk_test(self, x):
        y = x[x['action_type'] == 'clickout item']

        clk = y.tail(1)

        if len(y) == 1 and (x[pd.to_numeric(x['reference'], errors='coerce').notnull()].shape[0] == 0):
            # keep only sessions until last clickout
            x = x[x["step"] <= int(clk["step"])]

            if (x[x['action_type'] == 'clickout item'].shape[0] == 1) and (
                    x[pd.to_numeric(x['reference'], errors='coerce').notnull()].shape[0] == 0):
                return pd.DataFrame(x)

    def _fit(self, mode):
        """
        Cluster and predict for the test sessions without any numerical reference interactions

        self.train_indices: will contain all the train interactions
        self.test_indices: will contain all the test interactions
        self.target_indices: will contain the test interactions of sessions without
                            any other numerical reference interaction
        """
        # use only train of only cluster
        train = data.train_df(mode)
        train_groups = train.groupby(['session_id', 'user_id'], as_index=False).progress_apply(
            self.func_remove_steps_over_clk)

        self.train_indices = [x[1] for x in train_groups.index.values]

        # Those are groups of train I need, now let's keep only last clickout as part of the session

        test = data.test_df(mode)

        test_df = test.groupby(['session_id', 'user_id'])

        test_df = test_df.progress_apply(
            self.func_remove_steps_over_clk_test)

        if test_df.shape[0] > 0:
            self.target_indices = test_df[test_df.action_type == 'clickout item'].index.values
            # test_df has only those indices belonging to desired sessions cluster
            self.test_indices = list(list(zip(*test_df.index.values))[2])



if __name__ == '__main__':
    import utils.menu as menu

    obj = ClusterSessionsWithoutNumericalReferences()

    mode = menu.mode_selection()

    obj.save(mode)
