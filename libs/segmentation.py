import matplotlib.pyplot as plt
import numpy as np
import cv2
import histogram as hist
#import imutils

from scipy import ndimage
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.feature import canny
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.segmentation import morphological_geodesic_active_contour as morphocontour
from skimage.morphology import binary_closing
from skimage.filters import rank
from skimage.morphology import disk
from skimage.future import graph
from skimage.segmentation import clear_border
from skimage import exposure, restoration, img_as_float, morphology
from skimage.feature import peak_local_max
from skimage.restoration import estimate_sigma, denoise_nl_means
from myutils import imresize
from scipy.stats import mode
from timeit import default_timer as timer

def segmenta_test(img):

    segments_fz = felzenszwalb(img, scale=100, sigma=0, min_size=3)
    # segments_slic = slic(img, n_segments=500, compactness=10, sigma=1)
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

    print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))

    plt.figure()
    plt.imshow(mark_boundaries(img, segments_fz))
    plt.show()

def border_remove(img):

    edges = canny(img)
    #img[edges] = img[edges] - (img[edges])*0.1
    img[edges] = 255

    return img

def border_detection(img):

    edges = canny(img)
    fill_img = ndimage.binary_fill_holes(edges)
    label_objects, nb_labels = ndimage.label(fill_img)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 0
    mask_sizes[0] = 0
    img_cleaned = mask_sizes[label_objects]

    return img_cleaned

def region_based_segmentation(img):

    #img = border_remove(img)

    markers = np.zeros_like(img)

    markers[img <= 50] = 1
    markers[img > 150] = 2

    elevation_map = sobel(img)
    segmentation = watershed(elevation_map, markers, compactness=0.001)
    segmentation = ndimage.binary_fill_holes(segmentation - 1)
    segmentos, _ = ndimage.label(segmentation)

    print('Watershed Segmentation for no giants')
    plt.figure('Watershed Segmentation for no giants', figsize=(10, 8))
    plt.imshow(mark_boundaries(img, segmentos, color=(1, 0, 0)))
    plt.axis('off')
    plt.show()

    return segmentos, segmentation

def geo_segmentation(img):
    segmentos = morphocontour(img, 230, threshold=0.69)
    return segmentos

def segFelzenszwalb(sub_templateg, ragthreshold=35, scale=100, sigma=0, min_size=3):

    #print('Segmenting...')
    #sub_templateg = border_remove(sub_templateg)

    #s = timer()
    segmentos = felzenszwalb(sub_templateg, scale=scale, sigma=sigma, min_size=min_size)
    n_zeros = np.count_nonzero(segmentos)
    if n_zeros > 0:
        #print('Computing RAG...')
        rag = graph.rag_mean_color(sub_templateg, segmentos, mode='distance')
        new_labels = graph.cut_threshold(segmentos, rag, ragthreshold)
        segmentos = clear_border(new_labels)
    #e = timer()
    #print('Segmentation Done! ', round((e - s) / 60, 3), ' min')
    # print('tempo Felzenszwalb', len(np.unique(segmentos)))

    print('Felzemwalb Segmentation for giants')
    plt.figure('Felzemwalb Segmentation for giants', figsize=(10, 8))
    plt.imshow(mark_boundaries(sub_templateg, segmentos, color=(1, 0, 0)))
    plt.axis('off')
    plt.show()

    return segmentos