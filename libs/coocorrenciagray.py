#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__="Mauro"

import numpy as np
import histogram as hist
from skimage.io import imread
from skimage.feature import greycomatrix, greycoprops
import math
 
def coocorrencia(image,d=1,theta=0):
    
    #image=imread(img, as_grey=True)
    h = hist.histogram(image.astype('int32'))
    Ntonscinza = len(h)

    descritoresMC = np.zeros((1,4)).astype('double')
    #greycomatrix(imagem, d, theta, niveis de cinza da imagem)
    # g = greycomatrix(image, d, theta, Ntonscinza,normed=True,symmetric=True)

    g = greycomatrix(image, [d], [theta], Ntonscinza)

    g[0,:] = 0
    g[:,0] = 0

    # Para mostrar a matriz de coocorrencia
    # print g[:,:,0,0]

    # somat =  (np.sum(g)).astype('double')
    #
    # #Matriz de coocorrencia normalizada
    # gn = (g/somat).astype('double')

    # descritoresMC[0][0] = greycoprops(g, 'ASM')
    descritoresMC[0][0] = np.round(greycoprops(g, 'energy'), 2)
    descritoresMC[0][1] = np.round(greycoprops(g, 'contrast'), 2)
    descritoresMC[0][2] = np.round(greycoprops(g, 'homogeneity'), 2)
    # descritoresMC[0][4] = greycoprops(g, 'correlation')
    descritoresMC[0][3] = np.round(greycoprops(g, 'dissimilarity'), 2)

    #return g[:,:,0,0], gn[:,:,0,0], descritoresMC
    return descritoresMC