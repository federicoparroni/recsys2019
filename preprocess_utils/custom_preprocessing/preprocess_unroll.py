import data
from tqdm import tqdm
import copy

"""
    given a dataframe t, unrolls it with the following logic:

        - if a session has more than one clickout, creates a num of sessions
        equal to the number of the clickouts.
        
        - interactions after the last clickouts are preserved but not replicated.
        
        - sessions without any clickout are preserved.
        
        - unrolled sessions are denoted with a number of underscores
        after userid and sessions
"""

"""
    thats the function to give to create_full_df in order to impose this
    preprocessing
"""
def unroll_custom_preprocess_function(original_train, original_test):
    original_test = original_test.drop([794769, 794770]).reset_index(drop=True)
    return unroll(original_train), unroll(original_test)

def unroll(t):
    # unroll
    idx = []
    suffix = []
    to_append_partial = []
    a_user = t.loc[0].user_id
    a_sess = t.loc[0].session_id
    count = -1
    to_append = []
    for r in tqdm(zip(t.user_id, t.session_id, t.action_type)):
        count += 1
        if r[0] == a_user and r[1] == a_sess:
            to_append.append(count)
            if r[2] == 'clickout item':
                to_append_partial += copy.deepcopy([to_append])
        else:
            for i in range(len(to_append_partial)-1):
                idx += to_append_partial[i]
                suffix += ['_'*(i+1) for j in to_append_partial[i]]
            idx += to_append
            suffix += ['' for j in to_append]
            a_user = t.loc[count].user_id
            a_sess = t.loc[count].session_id
            to_append = [count]
            to_append_partial = []
            if r[2] == 'clickout item':
                to_append_partial += copy.deepcopy([to_append])

    for i in range(len(to_append_partial)-1):
        idx += to_append_partial[i]
        suffix += ['_'*(i+1) for j in to_append_partial[i]]
    idx += to_append
    suffix += ['' for j in to_append]

    # modify name of users and sessions
    t_new = t.loc[idx]
    user_with_suffix = []
    sess_with_suffix = []
    count = 0
    for r in tqdm(zip(t_new.user_id, t_new.session_id)):
        user_with_suffix.append('{}{}'.format(r[0],suffix[count]))
        sess_with_suffix.append('{}{}'.format(r[1],suffix[count]))
        count += 1
    t_new['user_id'] = user_with_suffix
    t_new['session_id'] = sess_with_suffix
    return t_new
