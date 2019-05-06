import sys
import os
sys.path.append(os.getcwd())

import utils.menu as menu
from clusterize.cluster_recurrent import ClusterRecurrent
from extract_features.reference_position_in_next_clickout_impressions import ReferencePositionInNextClickoutImpressions
from extract_features.global_popularity import GlobalPopularity
import preprocess_utils.session2vec as sess2vec



if __name__ == "__main__":
        
    mode = menu.mode_selection()
    sess_length = int(input('Insert the desired sessions length, -1 to not to pad/truncate the sessions: '))
    cluster_name = 'cluster_recurrent'

    # create the cluster
    cluster_choice = menu.yesno_choice('Do you want to create the cluster?', lambda: True, lambda: False)
    if cluster_choice:
        print('Creating the cluster...')
        cluster = ClusterRecurrent()
        cluster.save(mode)
        print()

    only_test = False
    if mode == 'full':
        only_test = menu.yesno_choice('Do you want to create only the test dataset?', lambda: True, lambda: False)

    # create the features to join
    ref_pos_next_clk_feat = ReferencePositionInNextClickoutImpressions(mode=mode, cluster=cluster_name)
    ref_pos_next_clk_feat.save_feature()
    print()
    ref_glob_popul = GlobalPopularity(mode=mode, cluster=cluster_name)
    ref_glob_popul.save_feature()
    print()

    features = [ref_pos_next_clk_feat, ref_glob_popul]
    # create the tensors dataset
    print('Creating the dataset ({})...'.format(mode))
    sess2vec.create_dataset_for_classification(mode, cluster_name, pad_sessions_length=sess_length,
                                                add_item_features=False, features=features, only_test=only_test)


