#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com \
           guozhilingty@gmail.com
  @Copyright: go-hiroaki & Chokurei
  @License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(object):
    def __init__(self, des="Mean Absolute Error"):
        self.des = des

    def __repr__(self):
        return "MAE"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 5-d ndarray in [batch_size, n_samples, channels, img_rows, img_cols]
            y_pred : 5-d ndarray in [batch_size, n_samples, channels, img_rows, img_cols]
        return mean_absolute_error, smaller the better
        """
        return torch.mean(torch.abs(y_pred - y_true), axis=(2,3,4))

class RMSE(object):
    def __init__(self, des="Root Mean Square Error"):
        self.des = des

    def __repr__(self):
        return "RMSE"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 5-d ndarray in [batch_size, n_samples, channels, img_rows, img_cols]
            y_pred : 5-d ndarray in [batch_size, n_samples, channels, img_rows, img_cols]
        return root_mean_square_error, smaller the better
        """
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, axis=(2,3,4)))


class PSNR(object):
    def __init__(self, des="Peak Signal to Noise Ratio"):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim=1):
        """
        args:
            y_true : 5-d ndarray in [batch_size, n_samples, channels, img_rows, img_cols]
            y_pred : 5-d ndarray in [batch_size, n_samples, channels, img_rows, img_cols]
        return peak_signal_to_noise_ratio, larger the better
        """
        mse = torch.mean((y_pred - y_true) ** 2, axis=(2,3,4))
        return 10 * torch.log10(1 / mse + 1e-7)


class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''
    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        # if torch.max(y_pred) > 128:
        #     max_val = 255
        # else:
        #     max_val = 1

        # if torch.min(y_pred) < -0.5:
        #     min_val = -1
        # else:
        #     min_val = 0
        # L = max_val - min_val
        L = 255

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

class AE(object):
    """
    Modified from matlab : colorangle.m, MATLAB V2019b
    angle = acos(RGB1' * RGB2 / (norm(RGB1) * norm(RGB2)));
    angle = 180 / pi * angle;
    """
    def __init__(self, des='Average Angular Error'):
        self.des = des

    def __repr__(self):
        return "AE"
    
    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        return average AE, smaller the better
        """
        dotP = torch.sum(y_pred * y_true, dim=1)
        Norm_pred = torch.sqrt(torch.sum(y_pred * y_pred, dim=1))
        Norm_true = torch.sqrt(torch.sum(y_true * y_true, dim=1))
        ae = 180 / math.pi * torch.acos(dotP / (Norm_pred * Norm_true + 1e-6))
        return ae.mean(1).mean(1)


if __name__ == "__main__":

    for ch in [3, 1]:

        batch_size, img_row, img_col = 1, 224, 224
        y_true = torch.rand(batch_size, ch, img_row, img_col)
        
        noise = torch.zeros(y_true.size()).data.normal_(0, std=0.1)
        y_pred = y_true + noise
        
        for cuda in [False]:#, True]:
            if cuda:
                y_pred = y_pred.cuda()
                y_true = y_true.cuda()

            print('#'*20, 'Cuda : {} ; size : {}'.format(cuda, y_true.size()))

            metric = RMSE()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))

            metric = PSNR()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))

            metric = SSIM()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))

            metric = AE()
            acc = metric(y_pred, y_true).item()
            print("{} ==> {}".format(repr(metric), acc))
            