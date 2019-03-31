from clusterize.clusterize_base import ClusterizeBase
import data

class RemainingTargets(ClusterizeBase):

    """
        Given a set of cluster names, it creates another cluster that has target ids 
        equal to the target ids not covered by the other clusters but still to predict
        in the actual mode
        Train: full train
        Test: full test
        Targets: targets not in use by the given clusters
    """

    def __init__(self, clusters):
        """
        clusters: list of cluster names
        """
        super(RemainingTargets, self).__init__('remaining_targets')
        self.clusters = clusters

    def _fit(self, mode):
        self.test_indices = data.test_indices(mode)
        self.train_indices = data.train_indices(mode)

        total_target_set = set(data.target_indices(mode=mode, cluster=data.SPLIT_USED))
        covered_target_list = []
        for i in self.clusters:
            covered_target_list += list(data.target_indices(mode, cluster=i))
        covered_target_set = set(covered_target_list)
        self.target_indices = list(total_target_set - covered_target_set)

if __name__ == '__main__':
    mode = 'local'
    obj = RemainingTargets(['interaction_item_image_one_step_before_missing_clk'])
    obj.save('local')
