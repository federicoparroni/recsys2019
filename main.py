from recommenders.collaborative_filterning.itembased import CFItemBased
from recommenders.distance_based_recommender import DistanceBasedRecommender

model = CFItemBased(mode='local')
model.run()
