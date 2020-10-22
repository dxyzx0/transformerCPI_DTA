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
from torch.utils.data import Subset, random_split

from DataUtil import *
from model import *

import numpy as np
from sklearn.model_selection import KFold

from datetime import date
import timeit
import random
import os

if __name__ == "__main__":
    # SEED = 1
    # random.seed(SEED)
    # torch.manual_seed(SEED)

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...', flush=True)

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()
        # scaler = None
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
    n_layers = 12
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 16 
    lr = 1e-4
    weight_decay = 1e-4
    iteration = 300
    kernel_size = 9
    num_workers = 16

    """ model pipeline """
    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                      PositionwiseFeedforward, dropout)
    model = Predictor(encoder, decoder)

    model.to(device)
    model = nn.DataParallel(model)
    # model.load_state_dict("output/2020-09-29/metric=rmse.state_dict")

    trainer = Trainer(model, lr, weight_decay, scaler)
    tester = Tester(model)

    """Output files."""
    param_setting = "protein_dim={},atom_dim={},hid_dim={},n_layers={},n_heads={},pf_dim={},dropout={},batch={},lr={},weight_decay={},iteration={},kernel_size={}".format(
        protein_dim, atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, batch, lr, weight_decay, iteration,
        kernel_size)
    output_path = "output/" + str(date.today()) + "/"
    os.makedirs(output_path, exist_ok=True)
    file_AUCs = output_path + param_setting + '.out'
    file_model_rmse = output_path + "metric={}".format('rmse')
    file_model_pear = output_path + "metric={}".format('pearson')
    file_model_spear = output_path + "metric={}".format('spearman')
    file_model_f1 = output_path + "metric={}".format('f1')
    file_model_auc = output_path + "metric={}".format('auc')
    file_model_r_square = output_path + "metric={}".format('r_square')
    file_model_R2 = output_path + "metric={}".format('R2')

    AUC = ('Epoch\tTime(sec)\tloss_train\trmse\tpearson\tspearman\tf1\tauc\tr_square\tR2')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    """ set printing precision options """
    np.set_printoptions(precision=2)

    """Start training."""
    print('Training...', flush=True)
    print(AUC, flush=True)
    start = timeit.default_timer()
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=15, gamma=0.7)
    min_rmse = 10 ** 10
    max_pear = 0
    max_spear = 0
    max_f1 = 0
    max_auc = 0
    min_r_square = 10 ** 10
    max_R2 = 0

    """ K fold split dataset """
    kf = KFold(n_splits=5, shuffle=False)

    for epoch in range(iteration):
        rmse_list = []
        pear_list = []
        spear_list = []
        f1_list = []
        auc_list = []
        r_square_list = []
        R2_list = []

        for train_index, test_index in kf.split(range(len(dta_ds))):
            """ train test dataloader """
            train_dataset, test_dataset = Subset(dta_ds, train_index), Subset(dta_ds, test_index)
            train_dl = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
            test_dl = DataLoader(test_dataset, batch_size=batch, shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn)

            loss_train = trainer.train(train_dl, device=device)

            rmse, pear, spear, f1, auc, r_square, R2 = tester.test(test_dl, device=device, plot=True)

            end = timeit.default_timer()
            time = end - start

            AUCs = [epoch + 1, time, loss_train, rmse, pear, spear, f1, auc, r_square, R2]
            scheduler.step()
            tester.save_AUCs(AUCs, file_AUCs)

            rmse_list.append(rmse)
            pear_list.append(pear)
            spear_list.append(spear)
            f1_list.append(f1)
            auc_list.append(auc)
            r_square_list.append(r_square)
            R2_list.append(R2)

            print('\t'.join(map(lambda x: "{:.2f}".format(x), AUCs)), flush=True)


        rmse = np.mean(rmse_list)
        pear = np.mean(pear_list)
        spear = np.mean(spear_list)
        f1 = np.mean(f1_list)
        auc = np.mean(auc_list)
        r_square = np.mean(r_square_list)
        R2 = np.mean(R2_list)

        if min_rmse > rmse:
            tester.save_model(model, file_model_rmse)
            min_rmse = rmse
        if max_pear < pear:
            tester.save_model(model, file_model_pear)
            tester.save_model(model, file_model_R2)
            max_R2 = R2
            max_pear = pear
        if max_spear < spear:
            tester.save_model(model, file_model_spear)
            max_spear = spear
        if max_f1 < f1:
            tester.save_model(model, file_model_f1)
            max_f1 = f1
        if max_auc < auc:
            tester.save_model(model, file_model_auc)
            max_auc = auc
        if min_r_square > r_square:
            tester.save_model(model, file_model_r_square)
            max_r_square = r_square

    print("Best performance:", flush=True)
    print(AUC, flush=True)
    print("best", 0, min_rmse, max_pear, max_spear, max_f1, max_auc, min_r_square, max_R2, flush=True)
