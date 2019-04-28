from clusterize.clusterize_base import ClusterizeBase
import data


import os
os.chdir("recsys2019")
print(os.getcwd())


class NumericReferenceOneStepBeforeMissingClk(ClusterizeBase):

    """
    Cluster for sessions to predict that have a numeric reference before the missing clickout
    Train: full train
    Test: full test
    Targets: just the missing clickouts that have numeric reference before
    """

    def __init__(self):
        super(NumericReferenceOneStepBeforeMissingClk, self).__init__('numeric_reference_one_step_before_missing_clk')

    def RepresentsInt(self, s):
        try: 
            int(s)
            return True
        except ValueError:
            return False

    def existsIndex(self, df, idx):
        try:
            df.loc[idx]
            return True
        except KeyError:
            return False

    def _fit(self, mode):
        self.train_indices = data.train_df(mode).index.values
        
        df = data.test_df(mode)
        self.test_indices = df.index.values

        just_missing_refs = df[df['reference'].isnull()]
        just_missing_refs = just_missing_refs[just_missing_refs['action_type'] == 'clickout item']
        idx_last_ref_numeric = []
        for idx, row in just_missing_refs.iterrows():
            sess = row['session_id']
            i = 1
            while True:
                if not self.existsIndex(df, idx - i):
                    break
                prev_row = df.loc[idx - i]
                if prev_row['session_id'] != sess:
                    break
                if self.RepresentsInt(prev_row['reference']):
                    if i == 1:
                        idx_last_ref_numeric.append(idx)
                        break
                    else:
                        break
                else:
                    i += 1
        
        self.target_indices = idx_last_ref_numeric


if __name__ == '__main__':
    obj = NumericReferenceOneStepBeforeMissingClk()
    obj.save('small')
