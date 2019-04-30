from recommenders.LightGBM_LambdaRank import LightGBMRanker
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import data
from tqdm import tqdm


def compute_MRR(self, predictions):
    """
    compute the MRR mean reciprocal rank of some predictions
    it uses the mode parameter to know which handle to retrieve to compute the score

    :param mode: 'local' or 'small' say which train has been used
    :param predictions: session_id, ordered impressions_list
    :param verboose: if True print the MRR
    :return: MRR of the given predictions
    """
    assert (self.mode == 'local' or self.mode == 'small')

    train_df = data.train_df("full", cluster=self.cluster)

    target_indices, recs = zip(*predictions)
    target_indices = list(target_indices)
    correct_clickouts = train_df.loc[target_indices].reference.values
    len_rec = len(recs)

    RR = 0
    print("Calculating MRR (hoping for a 0.99)")
    for i in tqdm(range(len_rec)):
        correct_clickout = int(correct_clickouts[i])
        if correct_clickout in predictions[i][1]:
            rank_pos = recs[i].index(correct_clickout) + 1
            if rank_pos <= 25:
                RR += 1 / rank_pos

    MRR = RR / len_rec
    print(f'MRR: {MRR}')


c = 0.01
l = []
for i in range(50):
    rounded = round(c, 2)
    l.append(rounded)
    c += 0.01

d = 0.01
ls = []
for i in range(50):
    rounded = round(d, 2)
    ls.append(rounded)
    d += 0.02

@use_named_args([
    #Real(0.01, 0.3, name='learning_rate'),
    Categorical(l, name='learning_rate'),
    Integer(2, 6, name='max_depth'),
    Integer(50, 500, name='n_estimators'),
    #Real(0,1, name='reg_lambda'),
    #Real(0,1, name='reg_alpha')
    Categorical(ls, name='reg_lambda'),
    Categorical(ls, name='reg_alpha'),
])
def objective(learning_rate, max_depth, n_estimators, reg_lambda, reg_alpha):
    model = LightGBMRanker(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                           reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    print(f'Iteration parameters: '
          f' - learning_rate= {learning_rate}  - max_depth= {max_depth} - n_estimators= {n_estimators} - reg_lambda= {reg_lambda} - reg_alpha= {reg_alpha}')

    model.fit()
    recommendations = model.recommend_batch()
    MRR = model.compute_MRR(recommendations)
    return -MRR


space = [
    Categorical(l, name='learning_rate'),
    Integer(2, 6, name='max_depth'),
    Integer(50, 500, name='n_estimators'),
    Categorical(ls, name='reg_lambda'),
    Categorical(ls, name='reg_alpha'),
]

res_gp = gp_minimize(objective, space, n_calls=10, random_state=42, verbose=True)
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
