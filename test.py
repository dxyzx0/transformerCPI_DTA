# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/25 10:03
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import random_split

from DataUtil import *
from model import *

import numpy as np

from datetime import date
import timeit
import random
import os

if __name__ == "__main__":
    #SEED = 1
    #random.seed(SEED)
    #torch.manual_seed(SEED)

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...', flush=True)

        # Creates a GradScaler once at the beginning of training.
        #scaler= GradScaler()
        scaler = None
    else:
        device = torch.device('cpu')
        scaler = None
        print('The code uses CPU!!!', flush=True)

    """Load preprocessed data."""
    word2vec_model = Word2Vec.load("word2vec_30.model")

    dta_ds = DTADataset("data/input/converted_all_data_drop_RCX_PDB_le_2500.csv", "data/output/smiles_map_2500.pkl",
                        "data/output/protein_map_2500.pkl", word2vec_model)
    # dta_ds = DTADataset("data/input/converted_all_data_drop_RCX_PDB_10000.csv", "data/output/smiles_map.pkl",
    #                     "data/output/protein_map.pkl", word2vec_model)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 48 
    lr = 1e-4
    weight_decay = 1e-4
    kernel_size = 9
    num_workers = 16 

    """ train test split """
    ratio = 0.9
    train_length = int(np.ceil(len(dta_ds) * ratio))
    test_length = len(dta_ds) - train_length
    train_dataset, test_dataset = random_split(dta_ds, lengths=(train_length, test_length))

    """ train test dataloader """
    train_dl = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dl = DataLoader(test_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                      PositionwiseFeedforward, dropout)
    model = Predictor(encoder, decoder)

    model.load_state_dict(torch.load("output/2020-09-29/metric=rmse.state_dict"))
    model.to('cpu')

    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output/result/AUCs--lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=9,n_layer=3,batch=64,balance,lookaheadradam_rmse'+ '_test.txt'
    AUC = ('Time(sec)\trmse\tpearson\tspearman\tf1\tauc\tr_square\tR2')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    """Start testing."""
    print(AUC, flush=True)
    start = timeit.default_timer()
    rmse, pear, spear, f1, auc, r_square, R2 = tester.test(test_dl)
    end = timeit.default_timer()
    time = end - start

    AUCs = [time, rmse, pear, spear, f1, auc, r_square, R2]
    tester.save_AUCs(AUCs, file_AUCs)
    print('\t'.join(map(str, AUCs)), flush=True)
