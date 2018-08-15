import math
import numpy as np

from collections import defaultdict
from scipy.stats import multivariate_normal


def dist(loc1, loc2):
    lat1, long1 = loc1.lat, loc1.lng
    lat2, long2 = loc2.lat, loc2.lng
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)
    earth_radius = 6371
    return arc * earth_radius


class Location(object):
    def __init__(self, id, lat, lng, freq, center=-1):
        self.id = id
        self.lat = lat
        self.lng = lng
        self.freq = freq
        self.center = center


class Center(object):
    def __init__(self):
        self.locations = []
        self.total_freq = 0
        self.distribution = None
        self.mu = None
        self.cov = None
        self.lat = None
        self.lng = None

    def add(self, loc):
        self.locations.append(loc)
        self.total_freq += loc.freq

    def build_gaussian(self):
        coo_seq = []
        for loc in self.locations:
            for _ in range(int(loc.freq)):
                coo_seq.append(np.array([loc.lat, loc.lng]))
        coo_seq = np.array(coo_seq)
        self.mu = np.mean(coo_seq, axis=0)
        self.cov = np.cov(coo_seq.T)
        self.distribution = multivariate_normal(self.mu, self.cov, allow_singular=True)
        self.lat = self.mu[0]
        self.lng = self.mu[1]

    def pdf(self, x):
        return self.distribution.pdf(np.array([x.lat, x.lng]))


class MultiGaussianModel(object):
    def __init__(self, alpha=0.2, theta=0.02, dmax=15):
        self.alpha = alpha
        self.theta = theta
        self.dmax = dmax
        self.poi_coos = None
        self.center_list = None

    def build_user_check_in_profile(self, sparse_check_in_matrix):
        L = defaultdict(list)
        for (uid, lid), freq in sparse_check_in_matrix.items():
            lat, lng = self.poi_coos[lid]
            L[uid].append(Location(lid, lat, lng, freq))
        return L

    def discover_user_centers(self, Lu):
        center_min_freq = max(sum([loc.freq for loc in Lu]) * self.theta, 2)
        Lu.sort(key=lambda k: k.freq, reverse=True)
        center_list = []
        center_num = 0
        for i in range(len(Lu)):
            if Lu[i].center == -1:
                center_num += 1
                center = Center()
                center.add(Lu[i])
                Lu[i].center = center_num
                for j in range(i+1, len(Lu)):
                    if Lu[j].center == -1 and dist(Lu[i], Lu[j]) <= self.dmax:
                        Lu[j].center = center_num
                        center.add(Lu[j])
                if center.total_freq >= center_min_freq:
                    center_list.append(center)
        return center_list

    def multi_center_discovering(self, sparse_check_in_matrix, poi_coos):
        self.poi_coos = poi_coos
        L = self.build_user_check_in_profile(sparse_check_in_matrix)

        center_list = {}
        for uid in range(len(L)):
            center_list[uid] = self.discover_user_centers(L[uid])
            for cid in range(len(center_list[uid])):
                center_list[uid][cid].build_gaussian()
        self.center_list = center_list

    def predict(self, uid, lid):
        lat, lng = self.poi_coos[lid]
        l = Location(None, lat, lng, None)

        prob = 0.0
        if uid in self.center_list:
            all_center_freq = sum([cid.total_freq**self.alpha for cid in self.center_list[uid]])
            all_center_pdf = sum([cid.pdf(l) for cid in self.center_list[uid]])
            if not all_center_pdf == 0:
                for cu in self.center_list[uid]:
                    prob += (
                        1.0 / (dist(l, cu) + 1) *
                        (cu.total_freq**self.alpha) / all_center_freq *
                        cu.pdf(l) / all_center_pdf)
        return prob

