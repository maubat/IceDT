import sys
sys.path.append('./libs')

import numpy as np
import matplotlib.pyplot as plt
import cv2
import histogram as hist
import pickle
import PIL
import pandas as pd

from osgeo import gdal
from skimage import img_as_ubyte
from PIL import Image
from skimage.filters import rank, median
from skimage.morphology import disk
from skimage.exposure import match_histograms, is_low_contrast, adjust_sigmoid

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

def improve_scene(sub_templateg, alg=1, classify=False):
    
    ref = np.asarray(PIL.Image.fromarray(np.asarray(pd.read_pickle('./AuxFiles/refhist.pkl'))))

    if is_low_contrast(sub_templateg, 0.35):
        
        temp_img = np.copy(sub_templateg).astype(float)
        temp_img[temp_img <= 30] = np.nan
        temp_img_std = np.nanstd(temp_img)
        del temp_img
                
        if temp_img_std <= 30:
                    
            print('Scene is low contrasted, improving...')
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.7)
            sub_templateg = adjust_sigmoid(sub_templateg)
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.2)
                     
    hist1 = cv2.calcHist([sub_templateg], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([ref], [0], None, [256], [0, 256])
    sim = cv2.compareHist(hist1, hist2, 0)
       
    print('Pre-processing for segmentation, histogram score', round(sim,3), 'of 1.0')
    if sim < 0.3:
        
        if alg == 1:

            print('Computing large objects.')
            if classify:
                sub_templateg = hist.adjust_gamma(sub_templateg, gamma=3.0)
                sub_templateg = adjust_sigmoid(sub_templateg)
                sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.0)
            else:
                sub_templateg = hist.adjust_gamma(sub_templateg, gamma=2.0)

        if alg == 2:

            print('Computing small objects')
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.5)
    else:
        
        if sim < 0.85:

            print('Computing objects.')
            
            if alg==1:
                sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.5)
                sub_templateg = adjust_sigmoid(sub_templateg)
                sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.0)
            
            if alg==2:
                if not classify:
                    sub_templateg = adjust_sigmoid(sub_templateg)
                else:
                    sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.5)

        else:
            print('Computing objects.')
            if not classify:
                sub_templateg = match_histograms(sub_templateg, ref)
                sub_templateg = convert(sub_templateg, 0, 255, np.uint8)
            #if not classify and alg == 2:
                #sub_templateg = adjust_sigmoid(sub_templateg)
            
    return sub_templateg                


def improve_scenebkp(sub_templateg, alg=1):
    
    ref = np.asarray(PIL.Image.fromarray(np.asarray(pd.read_pickle('./AuxFiles/refhist.pkl'))))
    
    #if alg==1:
        #sub_templateg = cv2.fastNlMeansDenoising(sub_templateg,None,10,7,21)
    
    if is_low_contrast(sub_templateg, 0.35):
        
        temp_img = np.copy(sub_templateg).astype(float)
        temp_img[temp_img <= 30] = np.nan
        temp_img_std = np.nanstd(temp_img)
        del temp_img
                
        if temp_img_std <= 30:
                    
            print('Scene is low contrasted, improving...')
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.7)
            sub_templateg = adjust_sigmoid(sub_templateg)
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.2)
                     
    hist1 = cv2.calcHist([sub_templateg], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([ref], [0], None, [256], [0, 256])
    sim = cv2.compareHist(hist1, hist2, 0)
            
    if sim < 0.8:

        if alg == 1:

            print('Computing large objects.')
            sub_templateg = cv2.medianBlur(sub_templateg, 5)
            sub_templateg = cv2.bilateralFilter(sub_templateg, 3, 3, 3)
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=3.0)
            sub_templateg = adjust_sigmoid(sub_templateg)
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.0)

        if alg == 2:

            print('Computing small objects')
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.5)
    else:

        if alg ==1:
            print('Computing large objects.')
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.5)
            sub_templateg = adjust_sigmoid(sub_templateg)
            sub_templateg = hist.adjust_gamma(sub_templateg, gamma=1.0)
                    
        else:
            print('Computing small objects.')
            sub_templateg = match_histograms(sub_templateg, ref)
            sub_templateg = convert(sub_templateg, 0, 255, np.uint8)
            
    return sub_templateg
