#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

import os
import numpy as np
from math import log10
from datetime import datetime
from PIL import Image
import Imath
from skimage.transform import resize

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def reinhard_tonemap(tensor):
    """Reinhard et al. (2002) tone mapping."""

    tensor[tensor < 0] = 0
    return torch.pow(tensor / (1 + tensor), 1 / 2.2)


def psnr(input, target):
    """Computes peak signal-to-noise ratio."""
    
    return 10 * torch.log10(1 / F.mse_loss(input, target))


def create_montage(img_name, noise_type, save_path, source_t, denoised_t, amount):
    """Creates montage for easy comparison."""

    # fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    # fig.canvas.set_window_title(img_name.capitalize()[:-4])
    source_imgs = []
    denoised_imgs = []
    resized_imgs = []
    # Bring tensors to CPU
    for i in range(amount):
        source_imgs.append(tvF.to_pil_image(source_t[i].cpu().narrow(0, 0, 1)))
        denoised_imgs.append(tvF.to_pil_image(denoised_t[i].cpu()))
        resized_img = source_imgs[i].resize((256, 256), Image.BICUBIC)
        resized_imgs.append(resized_img.resize((512, 512), Image.BICUBIC))
    
    # source = tvF.to_pil_image(source_t)
    # denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    


    # Concatenate images horizontally
    concatenated_image = Image.new('L', (source_imgs[0].width * 4 + 15, source_imgs[0].height*3 + 15))
    for i in range(amount):
        concatenated_image.paste(denoised_imgs[i], (source_imgs[0].width * i, source_imgs[0].height * 2 + 10))
        concatenated_image.paste(source_imgs[i], (source_imgs[0].width * i, source_imgs[0].height + 5))
        concatenated_image.paste(resized_imgs[i], (source_imgs[0].width * i, 0))

    # # Build image montage
    # psnr_vals = [psnr(source_t, clean_t), psnr(denoised_t, clean_t)]
    # titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
    #           'Denoised: {:.2f} dB'.format(psnr_vals[1]),
    #           'Ground truth']
    # zipped = zip(titles, [source, denoised, clean])
    # for j, (title, img) in enumerate(zipped):
    #     ax[j].imshow(img)
    #     ax[j].set_title(title)
    #     ax[j].axis('off')

    # Open pop up window, if requested

    # Save to files
    fname = img_name
    concatenated_image.save(os.path.join(save_path, f'{fname}-{noise_type}-concatenated.png'))
    # source.save(os.path.join(save_path, f'{fname}-{noise_type}-noisy.png'))
    # denoised.save(os.path.join(save_path, f'{fname}-{noise_type}-denoised.png'))
    # fig.savefig(os.path.join(save_path, f'{fname}-{noise_type}-montage.png'), bbox_inches='tight')


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
