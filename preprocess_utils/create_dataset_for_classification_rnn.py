import sys
import os
sys.path.append(os.getcwd())

import utils.menu as menu

from clusterize.cluster_recurrent import ClusterRecurrent
from clusterize.cluster_recurrent_up_to_len6 import ClusterRecurrentUpToLen6

from extract_features.reference_position_in_next_clickout_impressions import ReferencePositionInNextClickoutImpressions
from extract_features.global_interactions_popularity import GlobalInteractionsPopularity
from extract_features.global_clickout_popularity import GlobalClickoutPopularity
from extract_features.reference_price_in_next_clickout import ReferencePriceInNextClickout
from extract_features.average_price_in_next_clickout import AveragePriceInNextClickout

import preprocess_utils.session2vec as sess2vec



if __name__ == "__main__":
        
    mode = menu.mode_selection()
    #cluster_name = 'cluster_recurrent'
    cluster = menu.single_choice('Which cluster?', ['cluster recurrent','cluster recurrent len <= 6'],
                                    callbacks=[lambda: ClusterRecurrent, lambda: ClusterRecurrentUpToLen6])
    c = cluster()

    # create the cluster
    cluster_choice = menu.yesno_choice('Do you want to create the cluster?', lambda: True, lambda: False)
    if cluster_choice:
        print('Creating the cluster...')
        c.save(mode)
        print()

    only_test = False
    if mode != 'small':
        only_test = menu.yesno_choice('Do you want to create only the test dataset?', lambda: True, lambda: False)
    
    sess_length = int(input('Insert the desired sessions length, -1 to not to pad/truncate the sessions: '))

    # create the features to join
    ref_pos_next_clk_feat = ReferencePositionInNextClickoutImpressions()
    ref_pos_next_clk_feat.save_feature()
    print()
    glo_click_pop = GlobalClickoutPopularity()
    glo_click_pop.save_feature()
    print()
    glob_int_pop = GlobalInteractionsPopularity()
    glob_int_pop.save_feature()
    print()
    ref_price_feat = ReferencePriceInNextClickout()
    ref_price_feat.save_feature()
    print()
    avg_price_feat = AveragePriceInNextClickout()
    avg_price_feat.save_feature()
    print()

    features = [ref_pos_next_clk_feat, glo_click_pop, glob_int_pop, ref_price_feat, avg_price_feat]
    # create the tensors dataset
    print('Creating the dataset ({})...'.format(mode))
    sess2vec.create_dataset_for_classification(mode, c.name, pad_sessions_length=sess_length,
                                                add_item_features=False, features=features, only_test=only_test)


