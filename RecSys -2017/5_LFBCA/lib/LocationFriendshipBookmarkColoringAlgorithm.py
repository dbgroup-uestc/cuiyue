import time
import numpy as np
from collections import deque
from numpy.linalg import norm


class LocationFriendshipBookmarkColoringAlgorithm(object):
    def __init__(self, alpha, beta, epsilon):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.rec_score = None

    def PPR(self, u, friends, sim):
        alpha = self.alpha
        epsilon = self.epsilon

        q = deque()
        q_val = {}
        q.append(u)
        q_val[u] = 1.0
        ppr = np.zeros(sim.shape[0])

        while q:
            i = q.popleft()
            w = q_val[i]
            del q_val[i]

            ppr[i] += alpha * w
            if w > epsilon:
                for j in friends[i]:
                    if j in q_val:
                        q_val[j] += (1 - alpha) * w * sim[i, j]
                    else:
                        q_val[j] = (1 - alpha) * w * sim[i, j]
                        q.append(j)
        return ppr

    def precompute_user_social_similarities(self, check_in_matrix, social_matrix):
        C = check_in_matrix

        ctime = time.time()
        print("Computing user similarities...", )

        user_sim = C.dot(C.T)
        norms = [norm(C[i]) for i in range(C.shape[0])]

        for i in range(C.shape[0]):
            user_sim[i][i] = 0.0
            for j in range(i+1, C.shape[0]):
                user_sim[i, j] /= (norms[i] * norms[j])
                user_sim[j, i] /= (norms[i] * norms[j])

        for uid in range(user_sim.shape[0]):
            if not sum(user_sim[uid]) == 0:
                user_sim[uid] /= sum(user_sim[uid])
        print("Done. Elapsed time:", time.time() - ctime, "s")

        ctime = time.time()
        print("Computing social similarities...", )
        social_sim = social_matrix
        for uid in range(social_sim.shape[0]):
            if not sum(social_sim[uid]) == 0:
                social_sim[uid] /= sum(social_sim[uid])
        print("Done. Elapsed time:", time.time() - ctime, "s")

        return self.beta * user_sim + (1 - self.beta) * social_sim

    def compute_ppr_for_all_users(self, sim):
        ctime = time.time()
        print("Computing PPR values for all users...", )
        edges = (sim > 0)
        friends = [np.where(edges[uid, :] > 0)[0] for uid in range(sim.shape[0])]
        all_ppr = [self.PPR(uid, friends, sim) for uid in range(sim.shape[0])]
        print("Done. Elapsed time:", time.time() - ctime, "s")
        return np.array(all_ppr)

    def precompute_rec_scores(self, check_in_matrix, social_matrix):
        sim = self.precompute_user_social_similarities(check_in_matrix, social_matrix)
        all_ppr = self.compute_ppr_for_all_users(sim)
        normalized_check_in_matrix = np.zeros(check_in_matrix.shape)
        for uid in range(normalized_check_in_matrix.shape[0]):
            normalized_check_in_matrix[uid, :] = check_in_matrix[uid, :] / np.sum(check_in_matrix[uid, :])

        ctime = time.time()
        print("Precomputing recommendation scores...", )
        for uid in range(all_ppr.shape[0]):
            all_ppr[uid, uid] = 0.0
        self.rec_score = all_ppr.dot(normalized_check_in_matrix)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving result...",)
        np.save(path + "rec_score", self.rec_score)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, i, j):
        return self.rec_score[i][j]
