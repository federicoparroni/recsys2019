import os
import sys
sys.path.append(os.getcwd())

import data
from recommenders.collaborative_filterning.itembased import CFItemBased
import out

mode = 'small'

model = CFItemBased(mode=mode)
model.fit(k=900, distance=model.SIM_COSINE, shrink=0, implicit=False)
print('sim matrix: ', model._sim_matrix.shape)
recommendations = model.recommend_batch()

# out.create_sub()
