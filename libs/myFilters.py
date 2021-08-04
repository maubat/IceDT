from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

from scipy.stats import variation

from utils import assert_window_size
from utils import assert_indices_in_range

COEF_VAR_DEFAULT = 0.01
CU_DEFAULT = 0.25

def lee_filter(img, size):

    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def tv_filter(img):

    img_output = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
    return img_output