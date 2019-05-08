from recommenders.hybrid.borda_hybrid import Borda_Hybrid
from recommenders.hybrid.Probabilistic_Hybrid import Probabilistic_Hybrid
from evaluate.SubEvaluator import SubEvaluator
#model = Probabilistic_Hybrid(mode='local')
#model.run()

evaluator = SubEvaluator('xgboostlocal.csv')
evaluator.run()
