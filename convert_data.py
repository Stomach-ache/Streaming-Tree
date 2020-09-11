from xclib.data import data_utils
import xclib.evaluation.xc_metrics as xc_metrics
import numpy as np

dataset = 'eurlex'
# Read file with features and labels
features, labels, num_samples, num_features, num_labels = data_utils.read_data('data/' + dataset +'/' + 'train.txt')

A, B = 0.55, 1.5
inv_propen = xc_metrics.compute_inv_propesity(labels, A, B)
np.savetxt('inv_prop.txt', inv_propen)

data_utils.write_sparse_file(features, "trn_X_Xf.txt")
data_utils.write_sparse_file(labels, "trn_X_Y.txt")

features, labels, num_samples, num_features, num_labels = data_utils.read_data('data/' + dataset +'/' + 'test.txt')
data_utils.write_sparse_file(features, "tst_X_Xf.txt")
data_utils.write_sparse_file(labels, "tst_X_Y.txt")
