from recommenders.collaborative_filterning.itembased import CFItemBased
from recommenders.distance_based_recommender import DistanceBasedRecommender
from recommenders.content_based import ContentBased

model = ContentBased(mode='full')
model.run()
