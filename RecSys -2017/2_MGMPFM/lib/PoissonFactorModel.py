import time
import math
import numpy as np


class PoissonFactorModel(object):
    def __init__(self, K=30, alpha=20.0, beta=0.2):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.U, self.L = None, None

    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L...",)
        np.save(path + "U", self.U)
        np.save(path + "L", self.L)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L...",)
        self.U = np.load(path + "U.npy")
        self.L = np.load(path + "L.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def train(self, sparse_check_in_matrix, max_iters=50, learning_rate=1e-4):
        ctime = time.time()
        print("Training PFM...", )

        alpha = self.alpha
        beta = self.beta
        K = self.K

        F = sparse_check_in_matrix
        M, N = sparse_check_in_matrix.shape
        U = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (M, K))) / K
        L = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (N, K))) / K

        F = F.tocoo()
        entry_index = list(zip(F.row, F.col))

        F = F.tocsr()
        F_dok = F.todok()

        tau = 10
        last_loss = float('Inf')
        for iters in range(max_iters):
            F_Y = F_dok.copy()
            for i, j in entry_index:
                F_Y[i, j] = 1.0 * F_dok[i, j] / U[i].dot(L[j]) - 1
            F_Y = F_Y.tocsr()

            learning_rate_k = learning_rate * tau / (tau + iters)
            U += learning_rate_k * (F_Y.dot(L) + (alpha - 1) / U - 1 / beta)
            L += learning_rate_k * ((F_Y.T).dot(U) + (alpha - 1) / L - 1 / beta)

            loss = 0.0
            for i, j in entry_index:
                loss += (F_dok[i, j] - U[i].dot(L[j]))**2

            print('Iteration:', iters,  'loss:', loss)

            if loss > last_loss:
                print("Early termination.")
                break
            last_loss = loss

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.U, self.L = U, L

    def predict(self, uid, lid, sigmoid=False):
        if sigmoid:
            return 1.0 / (1 + math.exp(-self.U[uid].dot(self.L[lid])))
        return self.U[uid].dot(self.L[lid])
