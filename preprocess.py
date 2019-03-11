# import ptvsd
# ptvsd.enable_attach(address=('1.2.3.4', 3000), redirect_output=True)
# ptvsd.wait_for_attach()

import data
import preprocess.create_matrices as create

train = data.train_df()
test = data.test_df()
<<<<<<< HEAD
accomodations = data.accomodations_id()
print('data loaded')
u, session_ids, dict_acc, hdl = create.urm(train, test, accomodations, local=False, save=False)
=======
accomodations = data.accomodations_df()
u, session_ids, dict_acc, hdl = create.urm(train, test, accomodations, local=False, save=False)

>>>>>>> master
print(u.shape)