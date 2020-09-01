import cv2
import numpy as np

def histogram(image):

    # n_bins = image.max() + 1
    n_bins = 256
    hist, _ = np.histogram(image, normed=False, bins=n_bins, range=(0, n_bins-1))
    return hist
    # return np.bincount(image.ravel())

def histogram_eq(image):

    n_bins = image.max() + 1
    hist, _ = np.histogram(image, normed=True, bins=n_bins, range=(0, n_bins-1))
    return hist

def histogram_levels(image):

    lvl_max = image.max()
    lvl_min = image.min()

    return lvl_max,lvl_min

def adjust_gamma(im, gamma=1.0):
    image = np.copy(im)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


