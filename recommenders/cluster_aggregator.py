from recommenders.recommender_base import RecommenderBase
import data

class ClusterAggregator(RecommenderBase):

    def __init__(self, recs, mode='full'):
        """
        It gets recommendations coming from different recommenders and 
        puts those together.
        NOTICE: - Recommendation ids should be overlapped
                - Recommendation ids should cover all the ids of the actual mode used

        recs: list of recommendations coming out from a recommender
        """
        super(ClusterAggregator, self).__init__(mode=mode, cluster='no_cluster', name='cluster_aggregator')

    def fit(self):
        all_recs = {}
        self.recs = []
        for e in recs:
            if i in all_recs:
                raise Exception('Recommendation ids shouldn\'t be overlapped')
            all_recs[e[0]] = e[1]
        target_ids = data.target_indices(self.mode, self.cluster)
        for i in target_ids:
            if i not in all_recs:
                raise Exception('Recommendation ids should cover all the ids of the actual mode used')
            self.recs.append((i, all_recs[i]))

    def recommend_batch(self):
        return self.recs

    def get_scores_batch(self):
        raise Exception('It doesn\'t make sense to hybrid an aggregations of clusters! Hybrid the single clusters instead')
