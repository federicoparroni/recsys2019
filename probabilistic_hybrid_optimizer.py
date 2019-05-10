from recommenders.hybrid.Probabilistic_Hybrid import Probabilistic_Hybrid
from recommenders.hybrid.borda_hybrid import Borda_Hybrid
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import data

param_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

"""

Optimizer intended for the Probabilistic_Hybrid class.

"""

@use_named_args([
    Categorical(param_list, name='min_price_based'),
    Categorical(param_list, name='location_subm'),
    Categorical(param_list, name='lazyUserRec'),
    Categorical(param_list, name='last_interaction'),
    Categorical(param_list, name='xgb14f0665')
])
def objective(min_price_based,location_subm,lazyUserRec, last_interaction, xgb14f0665):
    params = {
        'min_price_based': min_price_based,
        'location_subm':location_subm,
        'lazyUserRec': lazyUserRec,
        'last_interaction': last_interaction,
        'xgb14f0665': xgb14f0665
    }

    model = Borda_Hybrid(params, mode='local')
    model.fit()
    sub = model.recommend_batch()
    MRR = model.score_sub(sub)

    print(f'Iteration parameters: '
          f" - min_price_based= {min_price_based} - location_subm= {location_subm} - lazyUserRec= {lazyUserRec} - last_interaction= {last_interaction} - xgb14f0665= {xgb14f0665}")

    return -MRR


space = [
    Categorical(param_list, name='min_price_based'),
    Categorical(param_list, name='location_subm'),
    Categorical(param_list, name='lazyUserRec'),
    Categorical(param_list, name='last_interaction'),
    Categorical(param_list, name='xgb14f0665')
]

res_gp = gp_minimize(objective, space, n_calls=250, n_random_starts=10, random_state=17, verbose=True)

print("""Best parameters:
- min_price_based= %d
- location_subm= %d
- lazyUserRec= %d
- last_interaction= %d
- xgb14f0665= %d
"""% (res_gp.x[0], res_gp.x[1],
                res_gp.x[2], res_gp.x[3], res_gp.x[4]
      ))
