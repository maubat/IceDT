__author__ = 'Mau'

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from scipy import ndimage as nd


def gabor_filter(image):
    # orig = np.copy(image)
    image = power(image)
    v = np.zeros((1, 2))
    kernel = np.real(gabor_kernel(0.1, theta = np.pi/4))
    feats = compute_feats(image, kernel)
    v[0][0], v[0][1] = feats
    # v[0][0] = round(v[0][0],2)
    # v[0][1] = round(v[0][1],2)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(orig, cmap='gray')
    # title = "Mean: " + str(round(np.mean(orig),5)) + " Variance: "+str(round(np.var(orig),5))
    # ax[0].set_title(title)
    # ax[0].set_xticks(())
    # ax[0].set_yticks(())
    # ax[1].imshow(image, cmap='gray')
    # title2 = "Mean: "+ str(round(v[0][0],5)) + " Variance: " +  str(round(v[0][1],5))
    # ax[1].set_title(title2)
    # ax[1].set_xticks(())
    # ax[1].set_yticks(())
    # plt.show()
    return v

def compute_feats(image, kernel):

    filtered = nd.convolve(image, kernel, mode='wrap')
    # plt.figure()
    # plt.imshow(filtered, cmap='gray')
    # plt.show()
    feat1 = filtered.mean()
    feat2 = filtered.std()
    return feat1,feat2

def power(image):

    kernel = np.real(gabor_kernel(0.1, theta = np.pi))
    # Normalize images for better comparison.
    if image.std() == 0:
        desvio = 1
    else:
        desvio = image.std()
    image = (image - image.mean()) / desvio
    saida = np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
                    nd.convolve(image, np.imag(kernel), mode='wrap')**2)
    return saida