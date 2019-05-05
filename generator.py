import os
import math
import numpy as np
import pandas as pd
import keras
import utils.sparsedf as sparsedf
import time

class DataGenerator(keras.utils.Sequence):
    """
    This class is used to read a csv in chunks (with 'tot_rows' rows), in order to feed it to a keras model.
    A batch is thus made up of a certain number of rows read from the csv file.
    A sample (a session) is composed by a fixed number of rows ('sample_size').
    """
    def __init__(self, dataset, for_train=True, samples_per_batch=256, #shuffle=True,
                pre_fit_fn=None, skip_rows=0, rows_to_read=None, read_chunks=False):
        """
        dataset_path (str): path to the folder containing the dataset files
        for_train (bool): init a generator for the training if True, otherwise for testing
        tot_rows (int): total number of rows to read from the csv
        rows_per_sample (int): number of rows per sample
        rows_to_read (int): number of rows to load for each epoch, None to load until the end
            #shuffle (bool): whether to shuffle or not the chunks in the batch
        pre_fit_fn (fn): function called before the batch of data is fed to the model (useful for some preprocessing)
                         (arg: Xchunk_df, Ychunk_df, index).
        skip_rows (int): number of rows to skip from the beginning of the file (useful for the validation generator)
        batches_per_epoch (int): number of batches to load at each epoch, -1 to load all samples
                        (useful to leave some batches for validation)
        read_chunks (bool): whether to read the dataset at chunks or to read the entire file once
        """
        self.dataset = dataset
        self.tot_rows = self.dataset.train_len if for_train else self.dataset.test_len
        assert skip_rows <= self.tot_rows
        
        self.samples_per_batch = samples_per_batch
        #self.shuffle = shuffle
        self.prefit_fn = pre_fit_fn
        self.for_train = for_train
        self.read_chunks = read_chunks

        # compute the total number of batches
        self.rows_per_batch = self.dataset.rows_per_sample * self.samples_per_batch
        
        if rows_to_read is not None and rows_to_read >= 0:
            # check if rows_to_read does not exceed the total rows to read
            assert rows_to_read + skip_rows <= self.tot_rows
        else:
            # set the number of rows to read equal to the remaining rows until the end
            rows_to_read = self.tot_rows - skip_rows
        self.rows_to_read = rows_to_read

        self.start_row_index = skip_rows
        self.end_row_index = self.start_row_index + rows_to_read
        self.start_batch_index = math.ceil(self.start_row_index / self.rows_per_batch)
        self.end_batch_index = math.ceil(self.end_row_index / self.rows_per_batch)
        self.start_session_index = math.ceil(self.start_row_index / self.dataset.rows_per_sample)
        self.end_session_index = math.ceil(self.end_row_index / self.dataset.rows_per_sample)
        
        self.tot_batches = math.ceil(self.rows_to_read / self.rows_per_batch)

        t0 = time.time()
        # load the entire dataset if read_chunks is False
        if not read_chunks:
            if self.for_train:
                self.dataset_X = self.dataset.load_Xtrain()
                self.dataset_Y = self.dataset.load_Ytrain()
            else:
                self.dataset_X, _ = self.dataset.load_Xtest()
        else:
            if self.for_train:
                self.dataset_X = pd.read_csv(self.dataset.X_train_path, index_col=0, skiprows=range(1, self.start_row_index+1), nrows=self.rows_to_read)
                self.dataset_Y = pd.read_csv(self.dataset.Y_train_path, index_col=0, skiprows=range(1, self.start_row_index+1), nrows=self.rows_to_read)
            else:
                self.dataset_X = pd.read_csv(self.dataset.X_test_path, index_col=0, skiprows=range(1, self.start_row_index+1), nrows=self.rows_to_read)
        
        #self.on_epoch_end()
        print(str(self))
        print('Preloading time: {}s\n'.format(time.time() - t0))

    def __len__(self):
        """ Return the number of batches that compose the entire dataset """
        #return int(np.ceil(len(self.x) / float(self.batch_size)))
        return self.tot_batches

    def __str__(self):
        lines = []
        lines.append('Dataset path: {} - mode: {}'.format(self.dataset.dataset_path, 'train' if self.for_train else 'test'))
        lines.append('One sample has {} row(s)'.format(self.dataset.rows_per_sample))
        lines.append('Reading from row {} (batch {}) to row {} (batch {})'.format(self.start_row_index,
                                        self.start_batch_index, self.end_row_index, self.end_batch_index))
        lines.append('Train has {} rows, {} samples'.format(self.dataset.train_len, 
                                        self.dataset.train_len / self.dataset.rows_per_sample))
        lines.append('Test has {} rows, {} samples'.format(self.dataset.test_len,
                                        self.dataset.test_len / self.dataset.rows_per_sample))
        lines.append('Samples per batch: {}'.format(self.samples_per_batch))
        lines.append('{} batch(es) will be read per epoch'.format(self.tot_batches))
        return '\n'.join(lines)

    def __getitem__(self, index):
        """ Generate and return one batch of data """
        #t0 = time.time()
        row_start = self.start_row_index + self.rows_per_batch * index
        row_end = min( row_start + self.rows_per_batch, self.end_row_index )

        if self.for_train:
            # return X and Y train
            if self.read_chunks:
                nrows = row_end - row_start
                Xchunk_df = pd.read_csv(self.dataset.X_train_path, index_col=0,
                                        skiprows=range(1, row_start+1), nrows=nrows)
                Ychunk_df = pd.read_csv(self.dataset.Y_train_path, index_col=0,
                                        skiprows=range(1, row_start+1), nrows=nrows)
            else:
                start_session_idx = self.start_session_index + self.samples_per_batch * index
                end_session_idx = min( start_session_idx + self.samples_per_batch, self.end_session_index )
                #print(' ', self.name, start_session_idx,'-',end_session_idx)

                Xchunk_df = self.dataset_X[start_session_idx : end_session_idx]
                Ychunk_df = self.dataset_Y[start_session_idx : end_session_idx]

            if callable(self.prefit_fn):
                out = self.prefit_fn(Xchunk_df, Ychunk_df, index)
            else:
                out = Xchunk_df.values, Ychunk_df.values
        else:
            #return only X test
            if self.read_chunks:
                nrows = row_end - row_start
                Xchunk_df = pd.read_csv(self.dataset.X_test_path, index_col=0,
                                        skiprows=range(1, row_start+1), nrows=nrows)
            else:
                start_session_idx = self.start_session_index + self.samples_per_batch * index
                end_session_idx = min( start_session_idx + self.samples_per_batch, self.end_session_index )
                
                Xchunk_df = self.dataset_X[start_session_idx : end_session_idx]
            
            if callable(self.prefit_fn):
                out = self.prefit_fn(Xchunk_df, index)
            else:
                out = Xchunk_df.values
        
        #print('Batch creation time: {}s\n'.format(time.time() - t0))
        return out
