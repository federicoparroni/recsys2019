import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from extract_features.feature_base import FeatureBase
import data
import numpy as np
import time
from preprocess_utils.last_clickout_indices import find

class TopPopInteractionFilters(FeatureBase):

    """
        preprocessing phase: This feature embeds some preprocessing:
            since any active current filter in session is dropped for action_types:

                'interaction item image', 'interaction item deals', 'interaction item info', 'interaction item rating',
                'change of sort order'

            I extended active filters in sessions also for those actions.
            With this information TopPopInteractionFilters generates a top pop on numeric interactions for sorting
            filters only, trying to cope with sessions without numeric references.

        clk_ref | Sort by Price | Sort by Distance | Sort by Rating | Best Value | Focus on Rating | .. | Sort by Popularity

        for every reference appeared in a numeric ref (not last clks) tells how many times this was interacted with
        a sorting filter

        Assumes that if nan is present in current_filters column or none of the sorting filters is active,
        Sort by Popularity is active.

        Added item_id that are not interacted: for those a row of zeros is joined

        """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'top_pop_interaction_sorting_filters'
        super(TopPopInteractionFilters, self).__init__(
            name=name, mode=mode, cluster=cluster)


    def extract_feature(self):

        def extend_session_current_filters(y):

            x = y
            cf = x.current_filters

            if len(cf.dropna()) == 0:
                return x

            ind = cf.dropna().head(1).index.values[0]  # indice del primo cf diverso da nan

            while ind < cf.tail(1).index.values[0]:  # serve un while per la fine della sessione

                Nan_ind_found = False
                nan_ind = ind

                while Nan_ind_found == False:  # trovo la prima action che annulla i filtri

                    if nan_ind == cf.tail(1).index.values[0]:
                        return x

                    try:
                        if x.loc[nan_ind + 1].action_type in ['interaction item image', 'interaction item deals',
                                                              'interaction item info', 'interaction item rating',
                                                              'change of sort order']:
                            nan_ind = nan_ind + 1
                            Nan_ind_found = True
                    except:
                        print(x)

                    else:

                        nan_ind += 1
                # ora nan_ind è l'indice della prima action che annulla i filtri
                Nan_ind_last_found = False
                not_nan_ind = nan_ind
                # scorro e modifico i valori di cf finchè non trovo il primo indice che nn annulla cf: not_nan_ind
                while Nan_ind_last_found == False:

                    cf.loc[not_nan_ind] = cf.loc[not_nan_ind - 1]

                    if not_nan_ind == cf.tail(1).index.values[0]:
                        x.current_filters = cf
                        return x

                    if x.loc[not_nan_ind + 1].action_type in ['search for poi', 'search for destination',
                                                              'search for item', 'filter selection', 'clickout item']:

                        not_nan_ind = not_nan_ind + 1
                        Nan_ind_last_found = True

                    else:

                        not_nan_ind += 1
                # ora not_nan_ind è il primo indice che non annulla cf (corrisponde a ind) si riparte da capo e si continua finchè
                # non si arriva a fine sessione
                ind = not_nan_ind

            x.current_filters = cf
            return x

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
        # extend current_filters
        df.groupby(['user_id', 'session_id']).progress_apply(extend_session_current_filters)
        indices_last_clks = find(df)
        d = df.drop(indices_last_clks)
        reference_rows = d[d.reference.astype(str).str.isnumeric()]
        d_splitted = reference_rows.current_filters.progress_apply(lambda x: str(x).split('|'))
        md = d_splitted.progress_apply(mask_sorting)
        df_f = df.loc[md.index]
        df_ref = df_f.reference
        dict_ref_to_filters = dict(
            zip(df_ref.unique(), [dict(zip(list_of_sorting_filters, np.zeros(len(list_of_sorting_filters)))) \
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

    c = TopPopInteractionFilters(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
