# -*- coding: utf-8 -*-
"""
Created on Tue May  9 01:39:16 2023

@author: kunyao
"""

'''
python test_fixed_prior.py \
--model_path logs/fp_epoch100_2_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/model.pth \
--log_dir logs/fp_epoch100_2_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/

python test_fixed_prior.py \
--model_path logs/fp_epoch100_no_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/model.pth \
--log_dir logs/fp_epoch100_no_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/

'''
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import finn_eval_seq, pred, plot_pred


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--data_root', default='../data', help='root directory for data')
parser.add_argument('--model_path', default='./logs/fp/cyclical-rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/model.pth', help='path to model')
parser.add_argument('--log_dir', default='cyclical_test', help='directory to save generations to')
parser.add_argument('--seed', default=2, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--num_workers', type=int, default=1, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=3, help='number of samples')
parser.add_argument('--N', type=int, default=3, help='number of samples')


args = parser.parse_args()
os.makedirs('%s' % args.log_dir, exist_ok=True)


args.n_eval = args.n_past+args.n_future
args.max_step = args.n_past + args.n_future

print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dtype = torch.cuda.FloatTensor



# ---------------- load the models  ----------------
modules = torch.load(args.model_path)
frame_predictor = modules['frame_predictor']
posterior = modules['posterior']
encoder = modules['encoder']
decoder = modules['decoder']

frame_predictor.batch_size = args.batch_size
posterior.batch_size = args.batch_size
args.g_dim = modules['args'].g_dim
args.z_dim = modules['args'].z_dim

# --------- transfer to gpu ------------------------------------
device = 'cuda'
frame_predictor.to(device)
posterior.to(device)
encoder.to(device)
decoder.to(device)

# ---------------- set the argsions ----------------
args.last_frame_skip = modules['args'].last_frame_skip

print(args)


# --------- load a dataset ------------------------------------
test_data = bair_robot_pushing_dataset(args, 'test')

test_loader = DataLoader(test_data,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)


if __name__ == '__main__':
    # plot test
    print("start test")
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    frame_predictor.eval()
    posterior.eval()
    encoder.eval()
    decoder.eval()
    for epoch in range(256):
        psnr_list = []
        for i, (test_seq, test_cond) in enumerate(tqdm(test_loader)):
            test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
            test_cond = test_cond.permute(1, 0, 2).to(device)
            pred_seq = pred(test_seq, test_cond, modules, args, device)
            _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
            psnr_list.append(psnr)
            
            
        ave_psnr = np.mean(np.concatenate(psnr_list))
        print(f'====================== test psnr = {ave_psnr:.5f} ========================')
        
        test_iterator = iter(test_loader)
        test_seq, test_cond = next(test_iterator)
        test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
        test_cond = test_cond.permute(1, 0, 2).to(device)
        plot_pred(test_seq, test_cond, modules, epoch, args)
        