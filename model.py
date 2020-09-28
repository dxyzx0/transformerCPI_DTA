# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/24 15:49
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast
import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from Radam import *
from lookahead import Lookahead

from eval_metrics import *


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(K.device)

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        # print(mask.shape)
        # print(energy.shape)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class Encoder(nn.Module):
    """protein feature extraction."""

    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        # self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        # pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        # protein = protein + self.pos_embedding(pos)
        # protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2*hid dim, protein len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, protein len]

            # apply residual connection / high way
            conved = (conved + conv_input) * self.scale.to(conv_input.device)
            # conved = [batch size, hid dim, protein len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        conved = self.ln(conved)
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout)
        self.ea = self_attention(hid_dim, n_heads, dropout)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""

    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.sa = self_attention(hid_dim, n_heads, dropout)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, 1)
        self.do_1 = nn.Dropout(0.2)
        # self.gn = nn.GroupNorm(8, 256)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        # norm = [batch size,compound len]
        norm: Tensor = F.softmax(norm, dim=1)
        # norm = [batch size,compound len]
        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)

        # sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        # for i in range(norm.shape[0]):
        #    for j in range(norm.shape[1]):
        #        v = trg[i, j, ]
        #        v = v * norm[i, j]
        #        sum[i, ] += v
        sum = torch.sum(trg * norm[:, :, None], axis=1)
        # sum = [batch size,hid_dim]

        label = self.do_1(F.relu(self.fc_1(sum)))
        # label = self.do_1(F.relu(self.fc_2(label)))
        label = self.do_1(F.relu(self.fc_2(label)))
        label = self.fc_3(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, Loss=nn.MSELoss(), atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.weight_1 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.weight_2 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()
        self.Loss = Loss

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight_1.size(1))
        self.weight_1.data.uniform_(-stdv, stdv)
        self.weight_2.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input =[batch,num_node, atom_dim]
        # adj = [batch,num_node, num_node]
        support = torch.matmul(input, self.weight_1)
        # support =[batch,num_node,atom_dim]
        output = torch.bmm(adj, support)
        # output = [batch,num_node,atom_dim]
        support = torch.matmul(output, self.weight_2)
        output = torch.bmm(adj, support)
        return output

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        # N = len(atom_num)  # batch size
        # compound_mask = torch.zeros((N, compound_max_len))
        # protein_mask = torch.zeros((N, protein_max_len))
        # for i in range(N):
        #    compound_mask[i, :atom_num[i]] = 1
        #    protein_mask[i, :protein_num[i]] = 1
        # compound_mask_1 = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        # protein_mask_1 = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        compound_axes = torch.arange(0, compound_max_len, device=atom_num.device).view(1, -1)
        compound_mask = (compound_axes < atom_num.view(-1, 1)).unsqueeze(1).unsqueeze(3)
        protein_axes = torch.arange(0, protein_max_len, device=protein_num.device).view(1, -1)
        protein_mask = (protein_axes < protein_num.view(-1, 1)).unsqueeze(1).unsqueeze(2)
        # print(torch.eq(compound_mask,compound_mask_1))
        # print(torch.eq(protein_mask,protein_mask_1))
        # print("compound:", compound_mask.shape)
        # print("Protein:", protein_mask.shape)
        # print("compound1_:", compound_mask.shape)
        # print("Protein1_:", protein_mask.shape)
        return compound_mask, protein_mask

    def forward(self, data):
        compound, adj, protein, correct_interaction, atom_num, protein_num = data
        # compound = [batch,atom_num, atom_dim]
        # adj = [batch,atom_num, atom_num]
        # protein = [batch,protein len, 100]

        compound_max_len = compound.shape[1]
        protein_max_len = protein.shape[1]
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)
        compound = self.gcn(compound, adj)
        # compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1 ,atom_num, atom_dim]

        # protein = torch.unsqueeze(protein, dim=0)
        # protein =[ batch size=1,protein len, protein_dim]
        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hid dim]

        predicted_interaction = self.decoder(compound, enc_src, compound_mask, protein_mask)
        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)
        loss = self.Loss(predicted_interaction, correct_interaction.view(-1, 1))
        return torch.unsqueeze(loss, 0), predicted_interaction.view(-1, 1), correct_interaction.view(-1, 1)


def to_cuda(data, device='cuda:0', cuda_available=True):
    compound, adj, protein, correct_interaction, atom_num, protein_num = data

    # Put input to cuda
    if cuda_available:
        compound = compound.to(device)
        adj = adj.to(device)
        protein = protein.to(device)
        atom_num = atom_num.to(device)
        protein_num = protein_num.to(device)
        correct_interaction = correct_interaction.to(device)

    return compound, adj, protein, correct_interaction, atom_num, protein_num


class Trainer(object):
    def __init__(self, model, lr, weight_decay, scaler=None):
        self.model = model
        self.scaler = scaler
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)

    def train(self, dataloader, device):
        self.model.train()

        loss_train = 0

        if self.scaler is None:
            for i, data_pack in enumerate(dataloader):
                data_pack = to_cuda(data_pack, device=device)

                loss, _, _ = self.model(data_pack)

                self.optimizer.zero_grad()
                loss.sum().backward()
                self.optimizer.step()

                loss_train += loss.sum().item()
        else:
            for i, data_pack in enumerate(dataloader):
                with autocast():
                    loss, _, _ = self.model(data_pack)

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss.sum()).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    loss_train += loss.sum().item()

        return loss.sum().item()


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, device, threshold=7., plot=False):
        self.model.eval()

        T, S = torch.Tensor(), torch.Tensor()
        with torch.no_grad():
            for i, data_pack in enumerate(dataloader):
                data_pack = to_cuda(data_pack, device=device)

                _, predicted_interaction, correct_interaction = self.model(data_pack)

                T = torch.cat((T, correct_interaction.cpu().detach()))
                S = torch.cat((S, predicted_interaction.cpu().detach()))

        T_ = T.squeeze().numpy()
        S_ = S.squeeze().numpy()

        if plot:
            np.savetxt('plot.csv', [T_, S_], delimiter=',')

        try:
            rmse = mean_squared_error(T_, S_)
            pear = pearson(T_, S_)
            spear = spearman(T_, S_)
            f1 = find_f1(T, S, threshold)
            auc = roc_auc_score(T_ > threshold, S_ > threshold)
            r_square = r2_score(T, S)
            R2 = pear ** 2
        except Exception as e:
            print(e)
            rmse = r_square = 10 ** 10
            pear = spear = f1 = auc = R2 = -1

        return rmse, pear, spear, f1, auc, r_square, R2

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename + ".state_dict")
        torch.save(model, filename + ".entire_model")
