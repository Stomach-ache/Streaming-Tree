#!/bin/bash

dataset="wiki10"
#dataset="eurlex"
#dataset="delicious"
#dataset="mirflickr"
data_dir="../sandbox/data/$dataset"
results_dir="../sandbox/results/$dataset"
model_dir="../sandbox/results/$dataset/model"

trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
trn_ft_lbl_file="${data_dir}/trn_X_XY.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
trn_score_file="${results_dir}/trn_score_mat.txt"
tst_score_file="${results_dir}/tst_score_mat.txt"
init_ratio=0.5
batch_size=3000
rand_seed=95

if [[ "$dataset" == "eurlex" ]]; then
    num_label=3993
elif [[ "$dataset" == "wiki10" ]]; then
    num_label=30938
elif [[ "$dataset" == "delicious" ]]; then
    num_label=983
elif [[ "$dataset" == "mirflickr" ]]; then
    num_label=38
fi

mkdir -p $model_dir

# training
# Reads training features (in $trn_ft_file), training labels (in $trn_lbl_file), and writes FastXML model to $model_dir

# NOTE: The usage of Bonsai for other datasets requires setting parameter `-m` to 2 for smaller datasets like EUR-Lex, Wikipedia-31K 
#       and to 3 for larger datasets like Delicious-200K, WikiLSHTC-325K, Amazon-670K, Wikipedia-500K, Amazon-3M.

#python gen_lbl_perm.py -md $model_dir -sz $num_label -sd $rand_seed

#'''
./bonsai_train $trn_ft_file $trn_lbl_file $trn_ft_lbl_file $tst_ft_file $model_dir $init_ratio $batch_size \
    -T 3 \
    -s 0 \
    -t 1 \
    -w 100 \
    -b 1 \
    -c 1.0 \
    -m 2 \
    -f 0 \
    -fcent 0 \
    -k 0.0001 \
    -siter 20 \
    -q 0 \
    -ptype 0 \
    -ctype 0
#'''

# testing on training set
#./bonsai_predict $trn_ft_file $trn_score_file $model_dir

# Reads test features (in $tst_ft_file), FastXML model (in $model_dir), and writes test label scores to $score_file
./bonsai_predict $tst_ft_file $tst_score_file $model_dir
#if [ ! -f ${score_file} ]; then
#    ./bonsai_predict $tst_ft_file $score_file $model_dir
#else
#    echo "using $score_file (cached)"
#fi

# performance evaluation
python get_all_metrics.py -Ytr $trn_lbl_file -Yte $tst_lbl_file -d $dataset -sc $tst_score_file -ir $init_ratio -bs $batch_size -md $model_dir
#matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('../tools')); trn_X_Y = read_text_mat('$trn_lbl_file'); tst_X_Y = read_text_mat('$tst_lbl_file'); wts = inv_propensity(trn_X_Y,0.55,1.5); score_mat = read_text_mat('$score_file'); get_all_metrics(score_mat, tst_X_Y, wts); exit;"
