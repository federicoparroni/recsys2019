import os
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
    def __init__(self, dataset, for_train=True, samples_per_batch=500, #shuffle=True,
                pre_fit_fn=None, skip_rows=0, batches_per_epoch=-1, read_chunks=False):
        """
        dataset_path (str): path to the folder containing the dataset files
        for_train (bool): init a generator for the training if True, otherwise for testing
        tot_rows (int): total number of rows to read from the csv
        rows_per_sample (int): number of rows per sample
        samples_per_batch (int): number of samples to load for each batch
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
        self.skip_rows = skip_rows
        self.batches_per_epoch = batches_per_epoch
        self.samples_per_batch = samples_per_batch
        #self.shuffle = shuffle
        self.prefit_fn = pre_fit_fn
        self.for_train = for_train
        self.read_chunks = read_chunks

        # compute the total number of batches
        self.rows_per_batch = self.dataset.rows_per_sample * self.samples_per_batch
        if self.batches_per_epoch > 0:
            self.tot_batches = batches_per_epoch
        else:
            self.tot_batches = int(np.ceil( (self.tot_rows - skip_rows) / self.rows_per_batch ))

        t0 = time.time()
        # load the entire dataset if read_chunks is False
        if not read_chunks:
            if self.for_train:
                self.dataset_X = self.dataset.load_Xtrain()
                self.dataset_Y = self.dataset.load_Ytrain()
            else:
                self.dataset_X = pd.read_csv(self.dataset.X_test_path, index_col=0, skiprows=range(1, self.skip_rows+1))
        
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
        lines.append('One sample has {} row(s), {} row(s) will be skipped'.format(
                                        self.dataset.rows_per_sample, self.skip_rows))
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
        if self.for_train:
            # return X and Y
            if self.read_chunks:
                row_start = index * self.rows_per_batch

                Xchunk_df = pd.read_csv(self.dataset.X_train_path, index_col=0,
                                        skiprows=range(1, self.skip_rows+row_start+1), nrows=self.rows_per_batch)
                Ychunk_df = pd.read_csv(self.dataset.Y_train_path, index_col=0,
                                        skiprows=range(1, self.skip_rows+row_start+1), nrows=self.rows_per_batch)
            else:
                start_batch_index = int(self.skip_rows / self.dataset.rows_per_sample) + self.samples_per_batch * index
                end_batch_index = min(start_batch_index + self.samples_per_batch, self.dataset.train_len)
                Xchunk_df = self.dataset_X[start_batch_index : end_batch_index]
                Ychunk_df = self.dataset_Y[start_batch_index : end_batch_index]

            if callable(self.prefit_fn):
                out = self.prefit_fn(Xchunk_df, Ychunk_df, index)
            else:
                out = Xchunk_df.values, Ychunk_df.values
        else:
            #return only X
            if self.read_chunks:
                row_start = index * self.rows_per_batch

                Xchunk_df = pd.read_csv(self.dataset.X_test_path, index_col=0,
                                        skiprows=range(1, self.skip_rows+row_start+1), nrows=self.rows_per_batch)
            else:
                start_batch_index = int(self.skip_rows / self.dataset.rows_per_sample) + self.samples_per_batch * index
                end_batch_index = min(start_batch_index + self.samples_per_batch, self.dataset.test_len)

                Xchunk_df = self.dataset_X[start_batch_index : end_batch_index]
            
            if callable(self.prefit_fn):
                out = self.prefit_fn(Xchunk_df, index)
            else:
                out = Xchunk_df.values
        
        #print('Batch creation time: {}s\n'.format(time.time() - t0))
        return out
