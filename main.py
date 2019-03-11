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
# urm = data.train_urm()
# handle = data.local_handle_df()
# accomodations = data.accomodations_df()
# dict_row = data.local_dictionary_row()
dict_col = data.dictionary_col()

#import preprocess.create_matrices as cm
#cm.urm_session_aware(data.local_train_df(),data.local_test_df(),'lin',False)

print('urm ', urm.shape)
model = DistanceBasedRecommender()
model._matrix_mul_order = 'inverse'
model.fit(urm, urm=urm, k=900, distance=model.SIM_COSINE, shrink=0, implicit=False)
print('sim matrix: ', model._sim_matrix.shape)
recommendations = model.recommend_only_target(handle, dict_row, dict_col)

out.create_sub(recommendations, handle)