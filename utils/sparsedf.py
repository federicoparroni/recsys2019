import os
import dask.dataframe as ddf


def read(path, sparse_cols=[], index_label=None, fill_nan_with=0):
    """ Load a sparse dataframe from a csv file. You can specify the sparse cols via sparse_cols. """
    def mappart(p, sparse_cols, fill_value):
        p[sparse_cols] = p[sparse_cols].to_sparse(fill_value=0)
        return p

    df = ddf.read_csv(path, blocksize=500e6)
    if len(sparse_cols) > 0:
        df = df.map_partitions(mappart, sparse_cols, fill_nan_with)
    df = df.compute()
    if index_label is not None and index_label != '':
        df = df.set_index(index_label)
    return df


def left_join_in_chunks(df1, df2, left_on, right_on, path_to_save, pre_join_fn=None, post_join_fn=None, chunk_size=int(10e6), data=dict(), index_label='orig_index'):
    # check and delete the file if already exists to avoid inconsistence in appending
    if os.path.isfile(path_to_save):
        os.remove(path_to_save)
    # join using chunks
    CHUNK_SIZE = chunk_size
    TOT_LEN = df1.shape[0]
    TOT = TOT_LEN + CHUNK_SIZE
    with open(path_to_save, 'a') as f:
        for i in range(0, TOT, CHUNK_SIZE):
            i_upper = max(i+CHUNK_SIZE, TOT_LEN)
            chunk_df = df1.iloc[i:i_upper]

            if callable(pre_join_fn):
                chunk_df, data = pre_join_fn(chunk_df, data)

            chunk_df = chunk_df.merge(df2, how='left', left_on=left_on, right_on=right_on)

            if callable(post_join_fn):
                chunk_df = post_join_fn(chunk_df, data)

            chunk_df.to_csv(f, header=f.tell() == 0, index_label=index_label, float_format='%.4f')
            print(f'{i_upper} / {TOT}', end='\r', flush=True)
    print()
    print('Done!')