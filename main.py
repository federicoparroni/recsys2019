from recommenders.collaborative_filterning.itembased import CFItemBased
from recommenders.distance_based_recommender import DistanceBasedRecommender
from recommenders.content_based import ContentBased
from validator import BayesianValidator

model = CFItemBased(mode='small', urm_name='urm_lin')
val = BayesianValidator(model)
val.validate(iterations=50)
