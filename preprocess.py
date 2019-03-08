import data
import preprocess.create_matrices as create

train = data.train_df()
test = data.test_df()
accomodations = data.accomodations_df()
u, session_ids, dict_acc, hdl = create.urm(train, test, accomodations['item_id'])

print(u.shape)