# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/25 10:03
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""

import torch
import numpy as np
import random
import os
import time
from model import *
import timeit
import pickle
import sys

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...', flush=True)
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!', flush=True)

    """Load preprocessed data."""


    with open("data/output/kinase_test.txt","rb") as f:
        data = pickle.load(f)

    dataset = shuffle_dataset(data, 1234)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-4
    weight_decay = 1e-4
    iteration = 300
    kernel_size = 9

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.load_state_dict(torch.load("output/model/lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=9,n_layer=3,batch=64,balance,lookaheadradam_rmse"))
    model.to(device)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output/result/AUCs--lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=9,n_layer=3,batch=64,balance,lookaheadradam_rmse'+ '_test.txt'
    AUC = ('Time(sec)\trmse\tpearson\tspearman\tf1\tauc\tr_square\tR2')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    """Start testing."""
    print(AUC, flush=True)
    start = timeit.default_timer()
    rmse, pear, spear, f1, auc, r_square, R2 = tester.test(dataset)
    end = timeit.default_timer()
    time = end - start

    AUCs = [time, rmse, pear, spear, f1, auc, r_square, R2]
    tester.save_AUCs(AUCs, file_AUCs)
    print('\t'.join(map(str, AUCs)), flush=True)
