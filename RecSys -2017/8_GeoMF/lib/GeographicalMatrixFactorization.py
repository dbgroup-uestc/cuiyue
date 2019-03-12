import time
from scipy.io import loadmat


class GeographicalMatrixFactorization(object):
    def __init__(self, ):
        self.P, self.Q = None, None
        self.X, self.Y = None, None

    def load_result(self, path):
        ctime = time.time()
        print("Loading result...",)
        mat = loadmat(path + 'GeoMF.mat')
        self.P = mat['P']
        self.Q = mat['Q']
        self.X = mat['X']
        self.Y = mat['Y']
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, u, l):
        return self.P[u, :].dot(self.Q[l, :]) + self.X[u, :].dot(self.Y[l, :])
