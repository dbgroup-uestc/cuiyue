import time
import numpy as np
from collections import defaultdict


class AdditiveMarkovChain(object):
    def __init__(self, delta_t, alpha):
        self.alpha = alpha
        self.delta_t = delta_t
        self.S = None
        self.OCount, self.TCount = None, None

    def build_location_location_transition_graph(self, sorted_training_check_ins):
        ctime = time.time()
        print("Building location-location transition graph (L2TG)...", )

        S = sorted_training_check_ins
        OCount = defaultdict(int)
        TCount = defaultdict(lambda: defaultdict(int))
        for u in S:
            last_l, last_t = S[u][0]
            for i in range(1, len(S[u])):
                l, t = S[u][i]
                if t - last_t <= self.delta_t:
                    OCount[last_l] += 1
                    TCount[last_l][l] += 1
                last_l, last_t = l, t

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.S = S
        self.OCount = OCount
        self.TCount = TCount

    def TP(self, l, next_l):
        if l not in self.OCount:
            return 1.0 if l == next_l else 0.0
        elif l in self.TCount and next_l in self.TCount[l]:
            return 1.0 * self.TCount[l][next_l] / self.OCount[l]
        else:
            return 0.0

    def W(self, i, n):
        return np.exp2(-self.alpha * (n - i))

    def predict(self, u, l):
        if u in self.S:
            n = len(self.S[u])
            numerator = np.sum([self.W(i, n) * self.TP(li, l) for i, (li, _) in enumerate(self.S[u])])
            denominator = np.sum([self.W(i, n) for i in range(len(self.S[u]))])
            return 1.0 * numerator / denominator
        return 1.0
