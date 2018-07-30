import numpy as np
from numpy.linalg import norm
import time


class UserBasedCF(object):
    def __init__(self):
        self.rec_score = None

    def load_result(self, path):
        ctime = time.time()
        print("Loading result...",)
        self.rec_score = np.load(path + "rec_score.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving result...",)
        np.save(path + "rec_score", self.rec_score)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def pre_compute_rec_scores(self, C):
        ctime = time.time()
        print("Training User-based Collaborative Filtering...", )
        # cosine similarity of user i and j
        sim = C.dot(C.T)
        norms = [norm(C[i]) for i in range(C.shape[0])]#norm the checkins record of user i

        for i in range(C.shape[0]):#0 axis: user
            sim[i][i] = 0.0# assign diagonal to be 0??? shouldn't it be 1???
            for j in range(i+1, C.shape[0]):
                sim[i][j] /= (norms[i] * norms[j])
                sim[j][i] /= (norms[i] * norms[j])

        self.rec_score = sim.dot(C)#w
        """
        for i in range(C.shape[0]):
            for j in range(i + 1, C.shape[0]):
                self.rec_score[i][j]=sum(sim[i]) #since in the later process there will normalize by user so this step is not necessary
        """
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, i, j):
        return self.rec_score[i][j]
