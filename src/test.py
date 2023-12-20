#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset, load_buffers
from noise2noise import Noise2Noise

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=4, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('-ts', '--train-size', help='size of train dataset', default = -1, type=int)

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)

    # Processing parameters
    parser.add_argument('-cr', '--random-crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('-cm', '--manual-crop-size', help='manual crop size', default=512, type=int)
    parser.add_argument('-rf', '--random-flip', help='random horizontal flip', default=0.0, type=float)

    return parser.parse_args()

if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.redux = False
    params.clean_targets = True
    buffers = load_buffers(100)
    test_loader = load_dataset(buffers, 1, params, shuffled=False, single=True)
    n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)
