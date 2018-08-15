import numpy as np
import scipy.sparse as sparse

from collections import defaultdict

from lib.PoissonFactorModel import PoissonFactorModel
from lib.MultiGaussianModel import MultiGaussianModel

from lib.metrics import precisionk, recallk


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
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))
    return sparse_training_matrix, training_tuples


def read_ground_truth():
    ground_truth = defaultdict(set)# value type is set
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def main():
    sparse_training_matrix, training_tuples = read_training_data()
    ground_truth = read_ground_truth()
    poi_coos = read_poi_coos()

    PFM.train(sparse_training_matrix, max_iters=10, learning_rate=1e-4)
    PFM.save_model("./tmp/")
    # PFM.load_model("./tmp/")
    MGM.multi_center_discovering(sparse_training_matrix, poi_coos)

    result_out = open("./result/aaai12_top_" + str(top_k) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    precision, recall = [], []
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            overall_scores = [PFM.predict(uid, lid) * MGM.predict(uid, lid)
                              if (uid, lid) not in training_tuples else -1
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
    poi_file = data_dir + "Gowalla_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100

    PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)
    MGM = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=15)

    main()
