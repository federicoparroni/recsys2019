import data
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import utils.check_folder as cf
import scipy.sparse as sps
import numpy as np


def create_ICM(name='icm.npz', save_path='dataset/matrices/full/'):
    """
    it creates the ICM matrix taking as input the 'item_metadata.csv'
    the matrix is saved in COO format to accomplish easy conversion to csr and csc
    a dictionary is also saved with key = item_id and values = row of icm containing the selected item

    :param name: name of the icm matrix
    :param save_path: saving path
    :param post_processing: post-processing functions to call on the newly created ICM
    :return:
    """
    print("creating ICM...\n")
    tqdm.pandas()
    attributes_df = data.accomodations_df()

    attributes_df['properties'] = attributes_df['properties'].progress_apply(
        lambda x: x.split('|') if isinstance(x, str) else x)
    attributes_df.fillna(value='', inplace=True)
    mlb = MultiLabelBinarizer()
    one_hot_attribute = mlb.fit_transform(attributes_df['properties'].values)
    one_hot_dataframe = pd.DataFrame(one_hot_attribute, columns=mlb.classes_)

    print("ICM created succesfully!\n")
    print("creating dictionary...\n")
    dict = {}
    item_ids = attributes_df['item_id'].values
    for i in tqdm(range(len(item_ids))):
        dict[item_ids[i]] = i

    print("saving ICM...\n")
    cf.check_folder(save_path)
    sps.save_npz(save_path + name, sps.coo_matrix(one_hot_dataframe.as_matrix()))

    print("saving dictionary")
    np.save(save_path + 'icm_dict.npy', dict)

    print("Procedure ended succesfully!")

if __name__ == '__main__':
    create_ICM()
