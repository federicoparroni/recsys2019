from preprocess_utils.last_clickout_indices import find
import data

"""
    return the indices of the test set in the actual mode 
    in which we are testing the algorithm, discarding all the part in which
    i'm actually training
"""
def retrieve_real_test_indices(mode, cluster):
    test = data.test_df(mode, cluster)
    idxs = sorted(find(test))
    test_indices = []
    for i in idxs:
        to_append = [i]
        a_user = test.at[i, 'user_id']
        a_sess = test.at[i, 'session_id']
        j = i-1
        while j >= test.index.values[0]:
            try:
                new_user = test.at[j, 'user_id']
                new_sess = test.at[j, 'session_id']
                if new_user == a_user and new_sess == a_sess:
                    to_append.append(j)
                    j -= 1
                else:
                    break
            except:
                j -= 1
            
        j = i+1
        while j <= test.index.values[-1]:
            try:
                new_user = test.at[j, 'user_id']
                new_sess = test.at[j, 'session_id']
                if new_user == a_user and new_sess == a_sess:
                    to_append.append(j)
                    j += 1
                else:
                    break
            except:
                j += 1

        test_indices += to_append
    return sorted(test_indices)