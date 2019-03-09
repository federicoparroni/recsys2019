import os
import sys
sys.path.append(os.getcwd())

import data
from recommenders.distance_based_recommender import DistanceBasedRecommender
import out

urm = data.urm()
handle = data.handle_df()
accomodations = data.accomodations_df()
dict_row = data.dictionary_row()
dict_col = data.dictionary_row()

print('urm ', urm.shape)
model = DistanceBasedRecommender()
model._matrix_mul_order = 'inverse'
model.fit(urm, urm=urm, k=900, distance=model.SIM_COSINE, shrink=0, implicit=False)
print('sim matrix: ', model._sim_matrix.shape)
recommendations = model.recommend_batch(handle, dict_row, dict_col)

out.create_sub(recommendations, handle)