# -*- coding: utf-8 -*-

import numpy as np
import argparse

parser = argparse.ArgumentParser('gen_lbl_perm')
parser.add_argument('--size', '-sz', metavar='SIZE', type=int, default=10,
                    help='set the size of labels')

parser.add_argument('--seed', '-sd', metavar='SEED', type=int, default=95,
                    help='set random seed')

parser.add_argument('--model_dir', '-md', metavar='MODEL_DIR', type=str,
                                               help='model dir to read predictions')
args = parser.parse_args()

np.random.seed(args.seed)

shuffled_indices = np.random.permutation(args.size)

with open(args.model_dir + '/lbl_idx', 'w') as fp:
    for idx in shuffled_indices:
        fp.write(str(idx) + '\n')
