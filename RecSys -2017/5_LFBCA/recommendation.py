import numpy as np
from collections import defaultdict

from lib.LocationFriendshipBookmarkColoringAlgorithm import LocationFriendshipBookmarkColoringAlgorithm
from lib.metrics import precisionk, recallk


def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_matrix = np.zeros((user_num, user_num))
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_matrix[uid1, uid2] = 1.0
        social_matrix[uid2, uid1] = 1.0
    return social_matrix


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, poi_num))
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        training_matrix[uid, lid] = freq
    return training_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def main():
    training_matrix = read_training_data()
    social_matrix = read_friend_data()
    ground_truth = read_ground_truth()

    LFBCA.precompute_rec_scores(training_matrix, social_matrix)
    LFBCA.save_result("./tmp/")

    result_out = open("./result/gis13_wang_top_" + str(top_k) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    precision, recall = [], []
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            overall_scores = [LFBCA.predict(uid, lid)
                              if training_matrix[uid, lid] == 0 else -1
                              for lid in all_lids]
            overall_scores = np.array(overall_scores)

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

    LFBCA = LocationFriendshipBookmarkColoringAlgorithm(alpha=0.85, beta=0.7, epsilon=0.001)

    main()
