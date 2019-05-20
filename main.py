from recommenders.hybrid.hybrid_impression_scores import HybridImpressionScores

import numpy as np


#scores_1 = np.load('scores/tf_scores_softmax_6619.npy')
#scores_2 = np.load('scores/tf_ranking_predictions_pairwise_logistic_loss_learning_rate_0.05_train_batch_size_256_hidden_layers_dim_256_128_num_train_steps_None_dropout_0.2_global_steps_15000_mrr_0.6605.npy')
#scores_3 = np.load('scores/tf_ranking_predictions_pairwise_hinge_loss_learning_rate_0.02_train_batch_size_32_hidden_layers_dim_256_128_128_num_train_steps_None_dropout_0.3_global_steps_38000_mrr_0.660410463809967.npy')
#scores_4 = np.load('scores/tf_ranking_predictions_softmax_loss_learning_rate_0.04_train_batch_size_128_hidden_layers_dim_512_256_128_num_train_steps_None_dropout_0.2_global_steps_27000_mrr_0.6590780019760132.npy')
#scores_5 = np.load('scores/localscores_catboost.npy')
#scores_6 = np.load('scores/rnn_GRU_2layers_64units_2dense_wgt_class_055780_balanced.npy')
#scores_7 = np.load('scores/xgboost_ranker_mode=local_cluste.npy')
scores_8 = np.load('scores/xgboost_ranker_balanced.npy')
scores_9 = np.load('scores/tf_b.npy')

evaluator = HybridImpressionScores('local', 'no_cluster', [scores_8, scores_9], [0.5,0.5], normalization_mode='MAX_ROW')
evaluator.evaluate()
