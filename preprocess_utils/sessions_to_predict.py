import pandas as pd

def find(df):
    """ Find the sessions with at least a missing clickout.
    Return the sessions with at least a missing clickout and the sessions with no missing clickouts.
    This function can be used to delete/move (to the training dataset) the sessions not to be predicted.
    """
    have_some_missing_clickouts = (lambda g: len(g[g.reference.isnull()]) > 0)
    missing_clickout_sessions_df = df.groupby(['user_id','session_id']).filter(have_some_missing_clickouts)
    return missing_clickout_sessions_df, df.drop(missing_clickout_sessions_df.index)
