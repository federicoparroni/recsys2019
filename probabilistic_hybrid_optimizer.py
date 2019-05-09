from recommenders.hybrid.Probabilistic_Hybrid import Probabilistic_Hybrid
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import data

param_list = [0,1,2,3,4,5,6,7,8,9,10]

"""

Optimizer intended for the Probabilistic_Hybrid class.

"""

@use_named_args([
    Categorical(param_list, name='last_interaction'),
    Categorical(param_list, name='lazyUserRec'),
    Categorical(param_list, name='location_subm'),
    Categorical(param_list, name='min_price_based'),
    Categorical(param_list, name='xgboostlocal'),
    Categorical(param_list, name='RNN_local')
])
def objective(last_interaction, lazyUserRec, location_subm, min_price_based, xgboostlocal, RNN_local):
    params = {
        'last_interaction': last_interaction,
        'lazyUserRec': lazyUserRec,
        'location_subm': location_subm,
        'min_price_based': min_price_based,
        'xgboostlocal': xgboostlocal,
        'RNN_local': RNN_local
    }

    model = Probabilistic_Hybrid(params, mode='local')
    model.fit()
    sub = model.recommend_batch()
    MRR = model.score_sub(sub)

    print(f'Iteration parameters: '
          f' - last_interaction= {last_interaction}  - lazyUserRec= {lazyUserRec} - location_subm= {location_subm} - min_price_based= {min_price_based} - xgboostlocal= {xgboostlocal} - RNN_local= {RNN_local}')

    return -MRR


space = [
    Categorical(param_list, name='last_interaction'),
    Categorical(param_list, name='lazyUserRec'),
    Categorical(param_list, name='location_subm'),
    Categorical(param_list, name='min_price_based'),
    Categorical(param_list, name='xgboostlocal'),
    Categorical(param_list, name='RNN_local')
]

res_gp = gp_minimize(objective, space, n_calls=150, n_random_starts=1, random_state=3, verbose=True)

print("""Best parameters:
- last_interaction= %d
- lazyUserRec= %d
- location_subm= %d
- min_price_based= %d
- xgboostlocal= %d
- RNN_local= %d
"""% (res_gp.x[0], res_gp.x[1],
                res_gp.x[2], res_gp.x[3],
                res_gp.x[4], res_gp.x[5]
      ))
