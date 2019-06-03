import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import os
from extract_features.feature_base import FeatureBase
import data
import numpy as np
import time
from preprocess_utils.last_clickout_indices import find
os.chdir("/Users/Albo/Documents/GitHub/keyblade95/recsys2019")


class TopPopSortingFilters(FeatureBase):

    """
    clk_ref | Sort by Price | Sort by Distance | Sort by Rating | Best Value | Focus on Rating | .. | Sort by Popularity

    for every reference appeared in a clickout (not the last ones) tells how many times this was clicked with
    sorting filters

    Assumes that if nan is present in current_filters column or none of the sorting filters is active,
    Sort by Popularity is active.

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'top_pop_sorting_filters'
        super(TopPopSortingFilters, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        list_of_sorting_filters_wout_pop = ['Sort by Price', 'Sort by Distance', 'Sort by Rating', 'Best Value',
                                            'Focus on Rating', 'Focus on Distance']

        list_of_sorting_filters = ['Sort by Price', 'Sort by Distance', 'Sort by Rating', 'Best Value',
                                   'Focus on Rating', 'Focus on Distance', 'Sort by Popularity']

        def mask_sorting(x):
            if np.isin(x, list_of_sorting_filters_wout_pop).any():
                return x
            else:
                return ['Sort by Popularity']


        start = time.time()
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        indices_last_clks = find(df)
        d = df[df.action_type == 'clickout item'].drop(indices_last_clks)
        d_splitted = d.current_filters.progress_apply(lambda x: str(x).split('|'))
        md = d_splitted.progress_apply(mask_sorting)
        df_f = df.loc[md.index]
        df_ref = df_f.reference
        dict_ref_to_filters = dict(
            zip(df_ref.unique(), [dict(zip(list_of_sorting_filters, np.zeros(len(list_of_sorting_filters))))\
                                     for i in range(len(df_ref.unique()))]))

        for index, row in tqdm(df_f.iterrows(), total=df_f.shape[0]):
            for i in md.loc[index]:
                if i in list_of_sorting_filters:
                    dict_ref_to_filters[row.reference][i] += 1
        df_feature = pd.DataFrame.from_dict(dict_ref_to_filters, orient='index')

        _time = time.time() - start
        elapsed = time.strftime('%Mm %Ss', time.gmtime(_time))
        print(f"elapsed in: {elapsed}")
        return df_feature.astype(int).reset_index().rename(index=str, columns={"index": "clk_ref"})


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = TopPopSortingFilters(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
