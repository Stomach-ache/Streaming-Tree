from scipy.sparse import hstack, csr_matrix, find, csc_matrix, vstack
import numpy as np
from evaluation import *
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse.linalg import norm
import argparse
from xclib.data import data_utils

parser = argparse.ArgumentParser('get_all_metrics')
parser.add_argument('--dataset', '-d', metavar='DATASET', type=str, default='eurlex',
                                               help='choose dataset to preceed')
parser.add_argument('--trnYfile', '-Ytr', metavar='NUM_TREE', type=str,
                                               help='set num of tree each forest')
parser.add_argument('--tstYfile', '-Yte', metavar='MAX_DEPTH', type=str,
                                               help='set max depth of trees')
parser.add_argument('--score', '-sc', metavar='MAX_DEPTH', type=str,
                                               help='set max depth of trees')
parser.add_argument('--model_dir', '-md', metavar='MODEL_DIR', type=str,
                                               help='model dir to read predictions')
parser.add_argument('--batch_size', '-bs', metavar='BATCH_SIZE', type=int,
                                               help='set batch_size in streaming label learning')
parser.add_argument('--init_ratio', '-ir', metavar='INIT_RATIO', type=float,
                                               help='set initial ratio of labels for pretraining')


args = parser.parse_args()


def csr2list(M):
    row, col, _ = find(M)
    res = [[] for _ in range(M.shape[0])]
    for r, c in zip(row, col):
        res[r].append(c)
    return res


Ytr = data_utils.read_sparse_file(args.trnYfile, force_header=True)
Yte = data_utils.read_sparse_file(args.tstYfile, force_header=True)
prob = data_utils.read_sparse_file(args.score, force_header=True)

mlb = MultiLabelBinarizer(range(Yte.shape[1]), sparse_output=True)
targets = mlb.fit_transform(csr2list(Yte))
train_labels = csr2list(Ytr)
if args.dataset.startswith('WikiPedia'):
    a, b = 0.55, 0.1
elif args.dataset.startswith('Amazon-'):
    a, b = 0.6, 2.6
else:
    a, b = 0.55, 1.5

inv_w = get_inv_propensity(mlb.transform(train_labels), a, b)
file = open('inv_w.txt', 'w')
for i in range(len(inv_w)):
    file.write(str(inv_w[i]) + '\n')
file.close()

num_sample, topk = prob.shape[0], 5
res = np.zeros((num_sample, topk))
for i in range(num_sample):
    #y = np.argsort(prob[i].data * inv_w[prob[i].indices])[-topk:][::-1]
    y = np.argsort(prob[i].data)[-topk:][::-1]
    if len(y) < topk:
        y = np.array(list(y) + [0] * (topk-len(y)))
    res[i] = prob[i].indices[y]

print (Ytr.shape)
print (Yte.shape)
print (res.shape)
#res = np.array(csr2list(res))

print(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
print(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')
print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_w, mlb), get_psp_3(res, targets, inv_w, mlb), get_psp_5(res, targets, inv_w, mlb))
print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_w, mlb), get_psndcg_3(res, targets, inv_w, mlb), get_psndcg_5(res, targets, inv_w, mlb))


with open(args.model_dir + '/lbl_idx', 'r') as fp:
    lbl_idx = list(map(int, fp.readlines()))
#print (lbl_idx)

num_label = Yte.shape[1]
base_no = int(args.init_ratio * num_label)
batch_idx = 0
batch_size = args.batch_size
j = 0
avg_p1 = 0
#avg_correct_unvalid = 0
#tmp_targets = targets.copy()

while j + batch_size <= num_label + int(batch_size * 0.1):
    print ("===============")
    score_file = args.model_dir + "/score_mat_init_ratio_" + str(int(args.init_ratio * 100)) + "_batch_size_" + str(batch_size) + "_" + str(batch_idx)
    print (score_file)
    prob = data_utils.read_sparse_file(score_file, force_header=True)

    '''
    lft = j
    rgt = base_no if batch_idx == 0 else min(j + batch_size, num_label)
    active_lbl = set(lbl_idx[lft:rgt])

    valid_idx = []
    correct_unvalid = 0
    '''
    for i in range(num_sample):
        '''
        is_valid = False
        for k in tmp_targets[i].indices:
            if k in active_lbl:
                is_valid = True
                break
        if is_valid == False:
            if len(prob[i].data) == 0:
                correct_unvalid += 1
            continue

        valid_idx.append(i)

        if len(prob[i].data) == 0:
            res.append([lbl_idx[lft]]*topk)
        else:
        '''
        y = np.argsort(prob[i].data)[-topk:][::-1]
        if len(y) < topk:
            y = np.array(list(y) + [y[-1]] * (topk-len(y)))
        res[i] = prob[i].indices[y]

    '''
    print ("=======len of valid_idx: ", len(valid_idx))
    targets = tmp_targets[valid_idx]
    print (len(res), targets.shape)
    #print (res[0])
    '''
    if batch_idx > 0:
        avg_p1 += get_p_1(res, targets, mlb)
    #    avg_correct_unvalid += correct_unvalid / (num_sample - len(valid_idx))
    #print (f'Old class detection Acc: {correct_unvalid / (num_sample - len(valid_idx))}')
    print(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
    print(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')

    if batch_idx == 0:
        j += base_no
    else:
        j += batch_size
    batch_idx += 1

avg_p1 /= (batch_idx - 1)
#avg_correct_unvalid /= (batch_idx - 1)
print (f'======Average Precision@1: {avg_p1}')
#print (f'======Average Correct Unvalid: {avg_correct_unvalid}')

'''
print ('=======re-ranking============')
for i in range(num_sample):
    y = np.argsort(prob[i].data * inv_w[prob[i].indices])[-topk:][::-1]
    if len(y) < topk:
        y = np.array(list(y) + [0] * (topk-len(y)))
    res[i] = prob[i].indices[y]

print(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
print(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')
print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_w, mlb), get_psp_3(res, targets, inv_w, mlb), get_psp_5(res, targets, inv_w, mlb))
print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_w, mlb), get_psndcg_3(res, targets, inv_w, mlb), get_psndcg_5(res, targets, inv_w, mlb))
'''
'''
results = open('./results/' + dataset, 'w')

results.write(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
results.write('\n')
results.write(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')
results.write('\n')
results.write(f'PSPrecision@1,3,5: {get_psp_1(res, targets, inv_w, mlb)}, {get_psp_3(res, targets, inv_w, mlb)}, {get_psp_5(res, targets, inv_w, mlb)}')
results.write('\n')
results.write(f'PSnDCG@1,3,5: {get_psndcg_1(res, targets, inv_w, mlb)}, {get_psndcg_3(res, targets, inv_w, mlb)}, {get_psndcg_5(res, targets, inv_w, mlb)}')
results.write('\n')
results.flush()
results.close()
'''
