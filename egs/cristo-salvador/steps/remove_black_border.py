#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import sys

from PIL import Image
from scipy.misc import imread, imshow
from skimage import exposure

def rlsa(x, th):
    H, W = x.shape
    z, f = 0, 1
    for i in xrange(H):
        for j in xrange(W):
            if x[i, j] == 1:
                if f == 1:
                    if z < th:
                        x[i, (j - z):j] = 1
                    else:
                        f = 0
                    z = 0
                f = 1
            else:
                if f == 1:
                    z += 1
    return x

def get_black_borders(x, th):
    H, W = x.shape
    horiz_hist = np.sum(x, 0) / H
    # count black frames from the left
    i = 0
    while i < W and horiz_hist[i] > 0:
        i += 1
    # count black frames from the right
    j = 0
    while j < W and horiz_hist[W - j - 1] > 0:
        j += 1
    if i > th: i = 0
    if j > th: j = 0
    return i, W - j - 1

if __name__ == '__main__':
    x =  2 * imread(sys.argv[1], True, 'F') / 255 - 1
    x = exposure.equalize_hist(x)
    x[x > 0.5] = 1
    x[x < 0.5] = 0
    x = 1 - x
    H, W = x.shape

    # Vertial RLSA
    x = rlsa(x.transpose(), H / 3)
    # Horizontal RLSA
    x = rlsa(x.transpose(), H / 3)
    i, j = get_black_borders(x, H / 2)
    crop_x = i
    crop_w = j - i + 1
    print '%dx%d+%d+%d' % (crop_w, H, crop_x, 0)
