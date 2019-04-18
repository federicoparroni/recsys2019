import os
import preprocess_utils.session2vec as sess2vec
import numpy as np
import pandas as pd
import keras
import utils.dataset_io as datasetio

class DataGenerator(keras.utils.Sequence):
    """
    This class is used to read a csv in chunks (with 'tot_rows' rows), in order to feed it to a keras model.
    A batch is thus made up of a certain number of rows read from the csv file.
    A sample (a session) is composed by a fixed number of rows ('sample_size').
    """
    def __init__(self, dataset_path, for_train=True, samples_per_batch=500, shuffle=True,
                pre_fit_fn=None, skip_rows=0, batches_per_epoch=-1):
        """
        dataset_path (str): path to the folder containing the dataset files
        for_train (bool): init a generator for the training if True, otherwise for testing
        tot_rows (int): total number of rows to read from the csv
        rows_per_sample (int): number of rows per sample
        samples_per_batch (int): number of samples to load for each batch
        shuffle (bool): whether to shuffle or not the chunks in the batch
        pre_fit_fn (fn): function called before the batch of data is fed to the model (arg: Xchunk_df, Ychunk_df, index).
        skip_rows (int): number of rows to skip from the beginning of the file (useful for the validation generator)
        batches_per_epoch (int): number of batches to load at each epoch, -1 to load all samples (useful to leave some batches for validation)
        """
        config = datasetio.load_config(dataset_path)
        self.tot_rows = config.train_len if for_train else config.test_len
        self.skip_rows = skip_rows
        self.rows_per_sample = config.rows_per_sample
        self.samples_per_batch = samples_per_batch
        self.X_path = config.x_train_path
        self.Y_path = config.y_train_path
        self.Xtest_path = config.x_test_path
        self.shuffle = shuffle
        self.prefit_fn = pre_fit_fn
        self.for_train = for_train

        # compute the total number of batches
        self.rows_per_batch = self.rows_per_sample * self.samples_per_batch
        if batches_per_epoch > 0:
            self.tot_batches = batches_per_epoch
        else:
            self.tot_batches = int(np.ceil( (self.tot_rows - skip_rows) / self.rows_per_batch ))

        #self.on_epoch_end()

    def __len__(self):
        """ Return the number of batches that compose the entire dataset """
        #return int(np.ceil(len(self.x) / float(self.batch_size)))
        return self.tot_batches

    def __getitem__(self, index):
        """ Generate and return one batch of data """
        row_start = index * self.rows_per_batch

        if self.for_train:
            # return X and Y
            Xchunk_df = pd.read_csv(self.X_path, index_col=0, skiprows=range(1, self.skip_rows+row_start+1), nrows=self.rows_per_batch)
            Ychunk_df = pd.read_csv(self.Y_path, index_col=0, skiprows=range(1, self.skip_rows+row_start+1), nrows=self.rows_per_batch)

            if callable(self.prefit_fn):
                out = self.prefit_fn(Xchunk_df, Ychunk_df, index)
                return out
        
            return Xchunk_df.values, Ychunk_df.values
        else:
            #return only X
            Xchunk_df = pd.read_csv(self.Xtest_path, index_col=0, skiprows=range(1, self.skip_rows+row_start+1), nrows=self.rows_per_batch)
            if callable(self.prefit_fn):
                out = self.prefit_fn(Xchunk_df, index)
                return out
        
            return Xchunk_df.values
