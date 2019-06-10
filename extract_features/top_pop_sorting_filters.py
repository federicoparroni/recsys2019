import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from extract_features.feature_base import FeatureBase
import data
import numpy as np
import time
from preprocess_utils.last_clickout_indices import find



class TopPopSortingFilters(FeatureBase):

    """
    clk_ref | Sort by Price | Sort by Distance | Sort by Rating | Best Value | Focus on Rating | .. | Sort by Popularity

    for every reference appeared in a clickout (not the last ones) tells how many times this was clicked with
    sorting filters

    Assumes that if nan is present in current_filters column or none of the sorting filters is active,
    Sort by Popularity is active.

    Added item_ids not clicked :for those a row of zeros is joined

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
        df_feature = df_feature.astype(int).reset_index().rename(index=str, columns={"index": "item_id"})
        set_of_not_clicked_items = set(data.accomodations_df().item_id) - set(df_feature.item_id)
        extension = pd.DataFrame(data=sorted([i for i in set_of_not_clicked_items]), columns=['item_id'])
        extd = df_feature.append(extension, ignore_index=True, sort=True)
        f = extd.fillna(0).reset_index().drop(columns=['index'])
        feature = f[np.insert(f.columns[:-1].values, 0, f.columns[-1])].astype(int)

        _time = time.time() - start
        elapsed = time.strftime('%Mm %Ss', time.gmtime(_time))
        print(f"elapsed in: {elapsed}")
        return feature


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = TopPopSortingFilters(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature().sort_values(by='item_id'))
