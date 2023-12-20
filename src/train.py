#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset, load_buffers
from noise2noise import Noise2Noise
from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=5, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', default = -1, type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', default = -1, type=int)
    parser.add_argument('-bs', '--buffer-size', help= 'size of buffer', default = 100, type=int)
    parser.add_argument('--full', help='full size', action='store_true')

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=16, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Processing parameters
    parser.add_argument('-cr', '--random-crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('-cm', '--manual-crop-size', help='random crop size', default=512, type=int)
    parser.add_argument('-rf', '--random-flip', help='random horizontal flip', default=0.0, type=float)
    

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    buffers = load_buffers(params.buffer_size)
    buffers_1 = load_buffers(100)
    train_loader = load_dataset(buffers, 0, params)
    valid_loader = load_dataset(buffers_1, 1, params)
    test_loader = load_dataset(buffers_1, 1, params, shuffled=False, single=True, test=True)
    #test_loader = load_dataset(buffers, 0, params, single=True)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader, test_loader)

# python train.py --cuda -e 300 -ts 256 -vs 100 --report-interval 16 -b 8 -bs 500