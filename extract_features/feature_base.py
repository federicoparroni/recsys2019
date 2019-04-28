from abc import abstractmethod
from abc import ABC
from utils.check_folder import check_folder
import pandas as pd
from utils.menu import yesno_choice
import os
from sklearn.preprocessing import MultiLabelBinarizer

"""
extend this class and give an implementation to extract_feature to 
make available a new feature
"""


class FeatureBase(ABC):
    """ Base class to create a new feature from the original files and save it to a new csv """

    def __init__(self, mode, cluster='no_cluster', name='featurebase', columns_to_onehot=[], save_index=False):
        """
        columns_to_onehot: [(columns_header, onehot_mode), ...]
            onehot_mode: 'single' or 'multiple'
                'single': if we have just one categorical value for row
                'multiple': if we have multiple ones (assumes pipe separation)

        eg: [('action', 'single')]
        meaning that the header of the column to onehot is 'action' and the onehot modality is 'single'
        
        save_index (bool): whether to save or not the index in the csv
        """
        self.mode = mode
        self.cluster = cluster
        self.name = name
        self.columns_to_onehot = columns_to_onehot
        self.save_index = save_index

    @abstractmethod
    def extract_feature(self):
        """
        Returns a dataframe that contains a feature (or more than one)
        on the first columns it should have an identifier of the single object to which the feature refers
        on the other column (or columns), the value of the features, with a meaningful name for the header.

        in case of categorical features, DO NOT RETURN A ONEHOT!
        In particular, return a single categorical value or a list of pipe-separated categorical values, and
        take care of setting self.columns_to_onehot nicely: base class will take care of one honetizing
        when read_feature is called.
        """
        pass

    def save_feature(self, overwrite_if_exists=None):
        """
        overwrite_if_exists: if true overwrite without asking; if false do not overwrite, if None ask before overwrite
        """
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
            self.cluster, self.mode, self.name)
        if os.path.exists(path):
            if overwrite_if_exists == None:
                choice = yesno_choice('feature exists yet. want to recreate?')
                if choice == 'n':
                    return
            elif not overwrite_if_exists:
                return
        df = self.extract_feature()
        check_folder(path)
        df.to_csv(path, index=self.save_index)


    def post_loading(self, df):
        """ Callback called after loading of the dataframe from csv. Override to provide some custom processings. """
        return df

    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe. The default implementation will join based on the
        common column between the 2 dataframes. Override to provide a custom join logic. """
        feature_df = self.read_feature(one_hot=one_hot)
        return df.merge(feature_df, how='left')

    def read_feature(self, one_hot=False):
        """
        it reads a feature from disk and returns it.
        if one_hot = True, it returns it as was saved.
        if one_hot = False, returns the onehot of the categorical columns, by means of self.columns_to_onehot
        """
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
            self.cluster, self.mode, self.name)
        if not os.path.exists(path):
            choice = yesno_choice('feature does not exist. want to create?')
            if choice == 'y':
                self.save_feature()
            else:
                return

        index_col = 0 if self.save_index else None
        df = pd.read_csv(path, index_col=index_col)
        #df = df.drop('Unnamed: 0', axis=1)

        print('{} feature read'.format(self.name))

        # then proceed with one hot
        if one_hot:
            for t in self.columns_to_onehot:
                col = df[t[0]]
                if t[1] == 'single':
                    oh = pd.get_dummies(col)
                else:
                    mid = col.apply(lambda x: x.split('|') if isinstance(x, str) else x)
                    mid.fillna(value='', inplace=True)
                    mlb = MultiLabelBinarizer()
                    oh = mlb.fit_transform(mid)
                    oh = pd.DataFrame(oh, columns=mlb.classes_)

                df = df.drop([t[0]], axis=1)
                df = df.join(oh)
            
            print('{} onehot completed'.format(self.name))

        df = self.post_loading(df)
        return df