import histogram as hist
import numpy as np
from scipy import stats
import getPercentis as gt

def get_mean(image):
    return np.nanmean(image)

def get_variance(image):
    return np.nanvar(image)

def get_desvpad(image):
    return np.nanstd(image)

def get_median(image):
    return np.nanmedian(image)

def get_mode(image):
    mode = stats.mode(image)
    return np.nanmax(mode)

def get_all_stats(imageb):
    
    image = np.copy(imageb).astype(float)
    image[image == 0.] = np.nan

    v = np.zeros((1,10)) # Inicializa o vetor de estatisticas
    v[0][0] = round(get_mean(image), 2)
    v[0][1] = round(get_desvpad(image), 2)
    v[0][2] = round(get_median(image), 2)
    v[0][3] = round(get_mode(image), 2)
    #p = [1, 10, 25, 75, 90, 99]
    #v[0][4:] = gt.image_percentil(image, p)

    return v
