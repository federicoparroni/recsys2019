from recommenders.catboost_rank import CatboostRanker

from recommenders.hybrid.hybrid_impression_scores import HybridImpressionScores
from recommenders.only_test_based.lazy_user_recommender import LazyUserRecommender
from recommenders.only_test_based.ordered_impression_based import OrderedImpressionRecommender

m1 = LazyUserRecommender(mode='small', cluster='no_cluster')
scores_1 = m1.get_scores_batch()

m2 = OrderedImpressionRecommender(mode='small', cluster='no_cluster')
scores_2 = m2.get_scores_batch()


evaluator = HybridImpressionScores('small', 'no_cluster', [scores_1, scores_2], [0.5, 0.5], normalization_mode='MAX_ROW')
evaluator.evaluate()
