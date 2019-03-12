import time
import numpy as np
import scipy.sparse as sparse


class TimeAwareMF(object):
    def __init__(self, K, Lambda, alpha, beta, T=24):
        self.K = K
        self.T = T
        self.Lambda = Lambda
        self.alpha = alpha
        self.beta = beta
        self.U = None
        self.L = None
        self.LT = None

    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L...",)
        for i in range(self.T):
            np.save(path + "U" + str(i), self.U[i])
        np.save(path + "L", self.L)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L...",)
        self.U = [np.load(path + "U%d.npy" % i) for i in range(self.T)]
        self.L = np.load(path + "L.npy")
        self.LT = self.L.T
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_sigma(self, path):
        ctime = time.time()
        print("Loading sigma...",)
        sigma = np.load(path + "sigma.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")
        return sigma

    def get_t_1(self, t):
        return (t - 1) if not t == 0 else (self.T - 1)

    def get_phi(self, C, i, t):
        t_1 = self.get_t_1(t)
        norm_t = np.linalg.norm(C[t][i, :].toarray(), 'fro')
        norm_t_1 = np.linalg.norm(C[t_1][i, :].toarray(), 'fro')
        if norm_t == 0 or norm_t_1 == 0:
            return 0.0
        return C[t][i, :].dot(C[t_1][i, :].T)[0, 0] / norm_t / norm_t_1

    def init_sigma(self, C, M, T):
        ctime = time.time()
        print("Initializing sigma...",)
        sigma = [np.zeros(M) for _ in range(T)]
        for t in range(T):
            C[t] = C[t].tocsr()
            for i in range(M):
                sigma[t][i] = self.get_phi(C, i, t)
        sigma = [sparse.dia_matrix(sigma_t) for sigma_t in sigma]
        print("Done. Elapsed time:", time.time() - ctime, "s")
        return sigma

    def train(self, sparse_check_in_matrices, max_iters=100, load_sigma=False):
        Lambda = self.Lambda
        alpha = self.alpha
        beta = self.beta
        T = self.T
        K = self.K

        C = sparse_check_in_matrices
        M, N = sparse_check_in_matrices[0].shape#usernum    locationnum

        if load_sigma:
            sigma = self.load_sigma("./tmp/")
        else:
            sigma = self.init_sigma(C, M, T)
            np.save("./tmp/sigma", sigma)

        U = [np.random.rand(M, K) for _ in range(T)]#U is of user and time
        L = np.random.rand(N, K)#L is of location it self

        C = [Ct.tocoo() for Ct in C]
        entry_index = [zip(C[t].row, C[t].col) for t in range(T)]

        C_est = [Ct for Ct in C]
        C = [Ct.tocsr() for Ct in C]

        for iters in range(max_iters):
            for t in range(T):
                C_est[t] = C_est[t].todok()
                for i, j in entry_index[t]:
                    C_est[t][i, j] = U[t][i].dot(L[j])
                C_est[t] = C_est[t].tocsr()

            for t in range(T):
                t_1 = self.get_t_1(t)
                numerator = C[t] * L + Lambda * sigma[t] * U[t_1]
                denominator = np.maximum(1e-6, C_est[t] * L + Lambda * sigma[t] * U[t_1] + alpha * U[t_1])
                U[t] *= np.sqrt(1.0 * numerator / denominator)

            numerator = np.sum([C[t].T * U[t] for t in range(T)], axis=0)
            denominator = np.maximum(1e-6, np.sum([C_est[t].T * U[t]], axis=0) + beta * L)
            L *= np.sqrt(1.0 * numerator / denominator)

            error = 0.0
            for t in range(T):
                C_est_dok = C_est[t].todok()
                C_dok = C[t].todok()
                for i, j in entry_index[t]:
                    error += (C_est_dok[i, j] - C_dok[i, j]) * (C_est_dok[i, j] - C_dok[i, j])
            print('Iteration:', iters, error)
        self.U, self.L = U, L
        self.LT = L.T

    def predict(self, i, j):
        return np.sum([self.U[t][i].dot(self.L[j]) for t in range(self.T)])

    def predict_all_pois_via_voting(self, u, top_k, visited_lids):
        candidates = []
        for t in range(self.T):
            candidates += list(reversed((self.U[t][u].dot(self.LT)).argsort()))[:top_k + len(visited_lids)]
        candidates_cnt = []
        for lid in set(candidates):
            if lid not in visited_lids:
                candidates_cnt.append([lid, candidates.count(lid)])
        candidates_cnt.sort(key=lambda k: k[1], reverse=True)
        return np.array(zip(*candidates_cnt[:top_k])[0])
