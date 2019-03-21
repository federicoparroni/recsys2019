from recommenders.collaborative_filterning.itembased import CFItemBased
from recommenders.distance_based_recommender import DistanceBasedRecommender
from recommenders.content_based import ContentBased

model = CFItemBased(mode='local', urm_name='urm_lin')
model.evaluate()
