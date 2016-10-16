#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import sys

from PIL import Image
from scipy.misc import imread, imshow
from scipy.ndimage.filters import gaussian_filter, uniform_filter
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


def get_borders(x, op, hist_th, width_th):
    H, W = x.shape
    horiz_hist = np.sum(x, 0) / H
    # count black frames from the left
    i = 0
    while i < W and op(horiz_hist[i], hist_th):
        i += 1
    # count black frames from the right
    j = 0
    while j < W and op(horiz_hist[W - j - 1], hist_th):
        j += 1
    if width_th is not None:
        if i > width_th: i = 0
        if j > width_th: j = 0
    return i, W - j - 1


def get_black_borders(x, hist_th, width_th):
    return get_borders(x, lambda x, t: x > t, hist_th, width_th)


def get_white_borders(x, hist_th, width_th):
    return get_borders(x, lambda x, t: x <= t, hist_th, width_th)


if __name__ == '__main__':
    # Read, equalize histogram and blur image to remove noise
    x = 1 - imread(sys.argv[1], True, 'F') / 255
    x = 2 * x - 1
    x = exposure.rescale_intensity(exposure.equalize_hist(x))
    x = exposure.rescale_intensity(uniform_filter(x, 5))
    H, W = x.shape

    # Binarize
    x[x >= 0.5] = 1
    x[x < 0.5] = 0

    # Vertial RLSA
    x = rlsa(x.transpose(), H / 3)
    # Horizontal RLSA
    x = rlsa(x.transpose(), H / 3)

    # Remove a few black pixels from the borders (width <= H / 2)
    i, j = get_black_borders(x, 0.0, H / 2)
    x = x[:,i:(j+1)]
    crop_x0 = i
    crop_x1 = j

    # Remove a few white pixels from the resulting borders
    i, j = get_white_borders(x, 0.0, None)
    x = x[:,i:(j+1)]
    crop_x0 = crop_x0 + i
    crop_x1 = crop_x0 + j

    print '%dx%d+%d+%d' % (crop_x1-crop_x0+1, H, crop_x0, 0)
