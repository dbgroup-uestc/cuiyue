import time
import math
import numpy as np

from collections import defaultdict


def dist(loc1, loc2):
    lat1, long1 = loc1
    lat2, long2 = loc2
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    earth_radius = 6371
    return arc * earth_radius


class FriendBasedCF(object):
    def __init__(self):
        self.social_proximity = defaultdict(list)
        self.sparse_check_in_matrix = None

    def compute_friend_sim(self, social_relations, poi_coos, sparse_check_in_matrix):
        self.sparse_check_in_matrix = sparse_check_in_matrix

        ctime = time.time()
        print("Precomputing similarity between friends...", )

        residence_lids = np.asarray(sparse_check_in_matrix.tocsr().argmax(axis=1)).reshape(-1)
        residence_coos = [poi_coos[lid] for lid in residence_lids.tolist()]
        max_distance = [-1.0 for _ in range(sparse_check_in_matrix.shape[0])]

        for uid1, uid2 in social_relations:
            dis = dist(residence_coos[uid1], residence_coos[uid2])
            max_distance[uid1] = max(max_distance[uid1], dis)
            max_distance[uid2] = max(max_distance[uid2], dis)
            self.social_proximity[uid1].append([uid2, dis])
            self.social_proximity[uid2].append([uid1, dis])

        for uid in self.social_proximity:
            # Max distance + 1 to smooth.
            self.social_proximity[uid] = [[fid, 1.0 - (dis / (1.0 + max_distance[uid]))]
                                          for fid, dis in self.social_proximity[uid]]
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, i, j):
        if i in self.social_proximity:
            numerator = np.sum([weight * self.sparse_check_in_matrix[k, j] for k, weight in self.social_proximity[i]])
            denominator = np.sum([weight for k, weight in self.social_proximity[i]])
            return numerator / denominator
        return 0.0
