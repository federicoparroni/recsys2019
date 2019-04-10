from recommenders.only_test_based.lazy_user_recommender import LazyUserRecommender

model = LazyUserRecommender(mode='local')
model.evaluate()

#val = BayesianValidator(model)
#val.validate(iterations=50)
