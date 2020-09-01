import numpy as np
import matplotlib.pyplot as plt
# import sys
import cv2
import gdal
import pickle

from skimage import img_as_ubyte
from PIL import Image
from skimage.filters import rank, median
from skimage.morphology import disk
from skimage.exposure import match_histograms

def convert(img, target_type_min, target_type_max, target_type):

    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)

    return new_img

def match_hist(img, ref):

    img = match_histograms(img, ref)
    img = convert(img, 0, 255, np.uint8)

    return img

def imresize(arr, size, interp='bilinear'):

    filters = {'bilinear': Image.BILINEAR,
               'nearest': Image.NEAREST,
               'bicubic': Image.BICUBIC,
               'antialias': Image.ANTIALIAS}

    l, c = arr.shape

    if type(size) == tuple:
        size_l, size_c = size
        size = (size_c, size_l)

    im = Image.fromarray(np.uint8(arr))

    if type(size) == tuple:
        imnew = im.resize(size, resample=filters[interp])
    else:
        size = (int(c*size), int(l*size))
        imnew = im.resize(size, resample=filters[interp])

    return np.array(imnew)


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return num, x
        num /= 1024.0

def fix_coast(mask_crop, img):

    lin, col = mask_crop.shape
    mask_crop_base = np.copy(mask_crop)
    xlin, xcol = np.where(mask_crop != 0)
    mask_crop_base[xlin, xcol] = 255

    mask_crop = imresize(mask_crop, 0.1)
    mask_crop = median(mask_crop, disk(30))
    mask_crop = imresize(mask_crop, (lin, col))

    xlin, xcol = np.where(mask_crop != 0)
    mask_crop[xlin, xcol] = 255
    mask_crop_f = mask_crop + mask_crop_base

    xlin, xcol = np.where(mask_crop_f != 0)
    land = img[xlin, xcol]

    land_pixels = [xlin, xcol, land]
    area_pixels = np.where(img > 0)

    return land_pixels, area_pixels

def pickle_masktif(mask, outfolder='./'):

    # Open mask tif file
    ds_mask = gdal.Open(mask)
    mask = np.asarray(ds_mask.GetRasterBand(1).ReadAsArray()).astype('uint8')
    mask = img_as_ubyte(mask)
    mask_gt = ds_mask.GetGeoTransform()

    file = open(outfolder+"mask.pkl", "wb")
    file2 = open(outfolder+"mask_gt.pkl", "wb")
    pickle.dump(mask, file)
    pickle.dump(mask_gt, file2)
    file.close()
    file2.close()
