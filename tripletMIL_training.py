# -*- coding: utf-8 -*-
import os
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score as auc_roc
import pickle
from arguments import parser, save_args
from utils import *
from data_utils import *
from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Net(nn.Module):
    def __init__(self, d, hidden_d):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d, hidden_d)
        self.out = nn.Linear(hidden_d, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


def get_scores(predictions, k_ratio):
    k_num = int(predictions.shape[0] * k_ratio)
    if k_num == 0:
        k_num = predictions.shape[0]
    top_scores = torch.topk(predictions[:, 0], k_num).values
    score_mean = torch.mean(top_scores)
    overall_score = score_mean
    return overall_score.unsqueeze(0)


def generate_data_loader(args, neg_bags, pos_bags):
    print('loading pos data')
    pos = MyDataset(pos_bags, isTest=False, sample_max_num=args.sample_max_num)
    print('loading neg data')
    neg = MyDataset(neg_bags, isTest=False, sample_max_num=args.sample_max_num)
    loader_anc = DataLoader(neg, batch_size=1, shuffle=True)
    loader_pos = DataLoader(pos, batch_size=1, shuffle=True)
    loader_neg = DataLoader(neg, batch_size=1, shuffle=True)
    return loader_anc, loader_pos, loader_neg


def evaluate(args, bags_ts, mlp, y_ts, trainval, auc_best=0):
    mlp.eval()
    test = MyDataset(bags_ts, isTest=True)
    loader_ts = DataLoader(test, batch_size=1)
    avg_predictions = []

    for param in mlp.parameters():
        param.requires_grad = False
    for idx_ts, tsbag in enumerate(loader_ts):
        tsbag = tsbag.float()
        tsbag = Variable(tsbag).type(torch.cuda.FloatTensor)
        scores = get_scores(mlp.forward(tsbag[0]), args.k_ratio)
        avg_predictions.append(float(scores))

    auc_avg = auc_roc(y_ts, avg_predictions)
    print(trainval, 'AUC=', auc_avg)

    if trainval == 'val':
        if auc_best < auc_avg:
            torch.save(mlp.state_dict(), os.path.join(args.output_dir, 'MLP.pth'))
            auc_best = auc_avg
    return auc_best, auc_avg


def main():
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    save_args(args)
    tensorboard_writer = SummaryWriter(log_dir=args.output_dir)

    patient_labels = read_patient_labels(args.patient_list)

    train_lst = read_split(os.path.join(args.split_path, 'train.txt'))
    val_lst = read_split(os.path.join(args.split_path, 'val.txt'))

    train_X, train_Y = read_features(train_lst, patient_labels, args.feature_dir)
    val_X, val_Y = read_features(val_lst,  patient_labels, args.feature_dir)
    
    pos_bags = train_X[train_Y == 1]
    neg_bags = train_X[train_Y == 0]

    print('pos bags:', len(pos_bags))
    print('neg bags:', len(neg_bags))

    mlp = Net(d=args.input_dim, hidden_d=args.hidden_dim)
    mlp.cuda()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    all_losses = []
    auc_best = -1
    mlp.train()
    n_iter = 0
    for e in range(args.epoch):
        loader_anc, loader_pos, loader_neg = generate_data_loader(args, neg_bags, pos_bags)
        print('Data loading done!')

        print('Epoch:', e)
        l = 0.0
        torch.autograd.set_detect_anomaly(True)

        for idx_p, pbag in enumerate(loader_pos):
            pbag = pbag.float()
            pbag = Variable(pbag, requires_grad=True).type(torch.cuda.FloatTensor)
            p_score = get_scores(mlp.forward(pbag[0]), args.k_ratio)

            loss_sum = -1
            for idx_n, nbag in enumerate(loader_neg):
                nbag = nbag.float()
                nbag = Variable(nbag, requires_grad=True).type(torch.cuda.FloatTensor)
                n_score = get_scores(mlp.forward(nbag[0]), args.k_ratio)

                for idx_anc, ancbag in enumerate(loader_anc):
                    ancbag = ancbag.float()
                    ancbag = Variable(ancbag, requires_grad=True).type(torch.cuda.FloatTensor)
                    anc_score = get_scores(mlp.forward(ancbag[0]), args.k_ratio)
                    z = np.array([0.0])
                    loss_p_n = torch.max(Variable(torch.from_numpy(z), requires_grad=True).type(torch.cuda.FloatTensor),
                                     (n_score - p_score + args.margin_inter))
                    loss_p_anc = torch.max(Variable(torch.from_numpy(z), requires_grad=True).type(torch.cuda.FloatTensor),
                                     (anc_score - p_score + args.margin_inter))
                    loss_n_anc = torch.max(Variable(torch.from_numpy(z), requires_grad=True).type(torch.cuda.FloatTensor),
                                     (torch.norm(anc_score - n_score, p=2) - args.margin_intra))
                    loss = loss_p_n + loss_p_anc + loss_n_anc

                    n_iter += 1
                    tensorboard_writer.add_scalar('loss', loss, global_step=n_iter)
                    if loss_sum == -1:
                        loss_sum = loss
                    else:
                        loss_sum += loss
                    break
            l = l + float(loss)
            print('sum_anc', round(float(anc_score), 3),
                  'sum_p', round(float(p_score), 3),
                  'sum_n', round(float(n_score), 3))
            optimizer.zero_grad()
            loss_sum.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=10.0)
            optimizer.step()
            # if idx_p == 3:
            #     break
            # print(idx_p, len(loader_pos))

        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
        print('Epoch Loss:', l)
        mlp.eval()
        auc_best, cur_auc = evaluate(args, val_X, mlp, val_Y, 'val', auc_best)
        tensorboard_writer.add_scalar('auc', cur_auc, global_step=n_iter)

        all_losses.append(l)
        if float(l) == 0.0:
            print('early stop!')
            break
        # break

    auc = evaluate(args, val_X, mlp, val_Y, 'val', auc_best)


if __name__ == '__main__':
    main()
