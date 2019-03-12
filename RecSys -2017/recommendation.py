import numpy as np
from collections import defaultdict

from lib.UserBasedCF import UserBasedCF
from lib.FriendBasedCF import FriendBasedCF
from lib.PowerLaw import PowerLaw

from lib.metrics import precisionk, recallk


def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_relations = defaultdict(list)
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_relations[uid1].append(uid2)
        social_relations[uid2].append(uid1)
    return social_relations


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, poi_num))
    for eachline in train_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        training_matrix[uid, lid] = 1.0
    return training_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


def main():
    training_matrix = read_training_data()
    social_relations = read_friend_data()
    ground_truth = read_ground_truth()
    poi_coos = read_poi_coos()

    U.pre_compute_rec_scores(training_matrix)
    # U.load_result("./tmp/")

    S.compute_friend_sim(social_relations, training_matrix)
    G.fit_distance_distribution(training_matrix, poi_coos)

    result_out = open("./result/sigir11_top_" + str(top_k) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    precision, recall = [], []
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            U_scores = normalize([U.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])
            S_scores = normalize([S.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])
            G_scores = normalize([G.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])

            U_scores = np.array(U_scores)
            S_scores = np.array(S_scores)
            G_scores = np.array(G_scores)

            overall_scores = (1.0 - alpha - beta) * U_scores + alpha * S_scores + beta * G_scores

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            precision.append(precisionk(actual, predicted[:10]))
            recall.append(recallk(actual, predicted[:10]))

            print(cnt, uid, "pre@10:", np.mean(precision), "rec@10:", np.mean(recall))
            result_out.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')


if __name__ == '__main__':
    data_dir = "../data/"

    size_file = data_dir + "Gowalla_data_size.txt"
    check_in_file = data_dir + "Gowalla_checkins.txt"
    train_file = data_dir + "Gowalla_train.txt"
    tune_file = data_dir + "Gowalla_tune.txt"
    test_file = data_dir + "Gowalla_test.txt"
    social_file = data_dir + "Gowalla_social_relations.txt"
    poi_file = data_dir + "Gowalla_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100
    alpha = 0.1
    beta = 0.1

    U = UserBasedCF()
    S = FriendBasedCF(eta=0.05)
    G = PowerLaw()

    main()
