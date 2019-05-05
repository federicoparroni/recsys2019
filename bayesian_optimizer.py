from recommenders.LightGBM_LambdaRank import LightGBMRanker
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import data



test_df = data.classification_test_df(
            mode='small', sparse=False, cluster='no_cluster', algo='lightGBM')
full = data.train_df('full')
test = data.test_df(mode='small', cluster='no_cluster')

@use_named_args([
    Real(0.01, 0.3, name='learning_rate'),
    #Categorical(l, name='learning_rate'),
    Integer(2, 6, name='max_depth'),
    Integer(50, 500, name='n_estimators'),
    Real(0,1, name='reg_lambda'),
    Real(0,1, name='reg_alpha')
    #Categorical(ls, name='reg_lambda'),
    #Categorical(ls, name='reg_alpha'),
])
def objective(learning_rate, max_depth, n_estimators, reg_lambda, reg_alpha):
    model = LightGBMRanker(class_test=test_df, test=test, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                           reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    print(f'Iteration parameters: '
          f' - learning_rate= {learning_rate}  - max_depth= {max_depth} - n_estimators= {n_estimators} - reg_lambda= {reg_lambda} - reg_alpha= {reg_alpha}')

    model.fit()
    recommendations = model.recommend_batch()
    MRR = model.compute_MRR(recommendations, full=full)
    return -MRR


space = [
    Real(0.01, 0.3, name='learning_rate'),
    Integer(2, 6, name='max_depth'),
    Integer(50, 500, name='n_estimators'),
    Real(0, 1, name='reg_lambda'),
    Real(0,1, name='reg_alpha'),
]

initial_parameters = [0.15, 5, 500, 1, 0.1]

res_gp = gp_minimize(objective, space, x0=initial_parameters, n_calls=60, n_random_starts=3, random_state=42, verbose=True)
print("""Best parameters:
- learning_rate= %.3f
- max_depth= %d
- n_estimators= %d
- reg_lambda= %.3f
- reg_alpha= %.3f
"""% (res_gp.x[0], res_gp.x[1],
                res_gp.x[2], res_gp.x[3],
                res_gp.x[4]
      ))
