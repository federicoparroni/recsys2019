import os
import math
from abc import ABC

import pandas as pd
import numpy as np
from functools import reduce
import data


class SubmissionClusterMerge(ABC):
    """
    class used to merge 2 or more submission into a submission for the ACM leaderboard
    :param filepaths: list of paths to get the files to merge.
    i.e. [ submission2.csv, submission3.csv]
    All submission are in the /recsys2019/submissions directory
    :param cluster : list of cluster to which each filepath belongs to
    i.e. [ no_cluster, cluster_session_no_num_ref]
    if no_cluster is in cluster parameter: the merge will be done in all the rows where no other cluster is active
    """

    def __init__(self, filepaths, cluster):
        self.filepaths = filepaths
        self.cluster = cluster

        self.target_sessions = list(data.test_df("full", "no_cluster")
                                    .iloc[data.target_indices("full", "no_cluster")].session_id.values)

        #TODO Check if filepaths exists

        self.absolute_path = 'submission/'



    def merge(self, submission_path):

        list_df = []
        if "cluster_sessions_no_numerical_reference" in self.cluster:
            idx = self.cluster.index("cluster_sessions_no_numerical_reference")
            sub_no_num = pd.read_csv(self.absolute_path + self.filepaths[idx])
            list_df.append(sub_no_num)

        if "numeric_reference_one_step_before_missing_clk" in self.cluster:
            idx = self.cluster.index("numeric_reference_one_step_before_missing_clk")
            sub_num = pd.read_csv(self.absolute_path + self.filepaths[idx])
            list_df.append(sub_num)

        if "other_cluster" in self.cluster:
            idx = self.cluster.index("other_cluster")
            sub_other = pd.read_csv(self.absolute_path + self.filepaths[idx])
            list_df.append(sub_other)

        #merge ->
        if len(list_df)>1:
            final = pd.concat(list_df)
        else:
            final = list_df[0]

        #Check if sessions covers all target indices or not
        if final.shape[0] < len(self.target_sessions):
            if "no_cluster" in self.cluster:
                #Means there are clusters not covering all target sessions
                #select all remaining target sessions

                print("Stucking rows from no_cluster submission...")
                def difference(list1, list2):
                    return list(set(list1) - set(list2))

                missing_sessions = difference(self.target_sessions, final.session_id.values)

                idx = self.cluster.index("no_cluster")
                filling_sub = pd.read_csv(self.absolute_path + self.filepaths[idx])
                filling_sub = filling_sub[filling_sub.session_id.isin(missing_sessions)]

                final = pd.concat([final, filling_sub])

        if final.shape[0] > len(self.target_sessions):
            print("shape of submission is bigger than expected! "
                  "Is exactly {} rows more".format(final.shape[0]-len(self.target_sessions)))
            return

        final.sort_values(by=['user_id'], inplace=True)

        final.to_csv(path_or_buf=self.absolute_path+submission_path, index=False)



if __name__ == "__main__":
    scm = SubmissionClusterMerge(['sub_no_cluster.csv', 'sub_no_num_ref.csv'], ['no_cluster', 'cluster_sessions_no_numerical_reference'])

    scm.merge('sub_vincente.csv')