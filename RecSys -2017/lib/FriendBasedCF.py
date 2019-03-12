import time
import numpy as np
from collections import defaultdict


class FriendBasedCF(object):
    def __init__(self, eta=0.5):
        self.eta = eta
        self.social_proximity = defaultdict(list)
        self.check_in_matrix = None

    def compute_friend_sim(self, social_relations, check_in_matrix):
        ctime = time.time()
        print("Precomputing similarity between friends...", )
        self.check_in_matrix = check_in_matrix
        for uid in social_relations:
            for fid in social_relations[uid]:
                if uid < fid:
                    u_social_neighbors = set(social_relations[uid])
                    f_social_neighbors = set(social_relations[fid])
                    jaccard_friend = (1.0 * len(u_social_neighbors.intersection(f_social_neighbors)) /
                                      len(u_social_neighbors.union(f_social_neighbors)))

                    u_check_in_neighbors = set(check_in_matrix[uid, :].nonzero()[0])
                    f_check_in_neighbors = set(check_in_matrix[fid, :].nonzero()[0])
                    jaccard_check_in = (1.0 * len(u_check_in_neighbors.intersection(f_check_in_neighbors)) /
                                        len(u_check_in_neighbors.union(f_check_in_neighbors)))
                    if jaccard_friend > 0 and jaccard_check_in > 0:
                        self.social_proximity[uid].append([fid, jaccard_friend, jaccard_check_in])
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, i, j):
        if i in self.social_proximity:
            numerator = np.sum([(self.eta * jf + (1 - self.eta) * jc) * self.check_in_matrix[k, j]
                                for k, jf, jc in self.social_proximity[i]])
            return numerator
        return 0.0
