import pandas as pd
from tqdm.auto import tqdm
import time
tqdm.pandas()
import os
os.chdir("/Users/Albo/Documents/GitHub/keyblade95/recsys2019")
from extract_features.feature_base import FeatureBase
import data


class StepsBeforeLastClickout(FeatureBase):
    """
    for every session the last steps considering some heuristic. Analysis of data and explanation on jupyter notebook
    tomorrow
    session_id | steps

    """
    def __init__(self, mode, cluster='no_cluster'):
        name = 'last_steps_before_clickout'
        super(StepsBeforeLastClickout, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):

            def last_important_steps(x):

                y = x[x.action_type == 'filter selection']
                z = x[x.action_type == 'clickout item']

                if len(z) > 0:
                    if (len(y) == 0) & (len(z) == 1):
                        from_ = x.head(1)
                    elif (len(y) == 0) & (len(z) > 1):
                        from_ = z.iloc[-2]
                    elif (len(y) > 0) & (len(z) == 1):
                        from_ = y.iloc[-1]
                    elif (len(y) > 0) & (len(z) > 1):
                        if int(y.iloc[-1].step) > int(z.iloc[-2].step):
                            from_ = y.iloc[-1]
                        else:
                            from_ = z.iloc[-2]
                    return int(x.tail(1).step) - int(from_.step)
                else:
                    return -1

            _important_steps = x.groupby('session_id').progress_apply(last_important_steps)
            session_last_steps = _important_steps[_important_steps != -1] + 1
            return pd.DataFrame(session_last_steps).reset_index().rename(columns={0: 'steps'})


        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = func(df)
        return s


if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = StepsBeforeLastClickout(mode=mode, cluster='no_cluster')
    c.save_feature()
    #print(c.read_feature())
