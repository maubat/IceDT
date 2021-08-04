# -*- encoding: utf-8 -*-
import sys
import numpy as np
import getPercentis as gt
import matplotlib.pyplot as plt

def histogram2stats(h):

    v = np.zeros((1,13))# Inicializa o vetor de estatisticas
    # Computa Estatisticas
    #Normaliza o histograma
    hn = 1.0 * h / h.sum()
    n = len(hn) # Numero de valores de cinza
    v[0][0] = round(np.sum((np.arange(n)*hn)),2) # mean
    v[0][1] = round(np.sum(np.power((np.arange(n)-v[0][0]),2)*hn)/n,2) # Variancia
    if v[0][1] == 0: v[0][1] = 0.1
    v[0][2] = round(np.sum(np.power((np.arange(n)-v[0][0]),3)*hn)/(np.power(v[0][1],1.5)),2)# skewness
    v[0][3] = round(np.sum(np.power((np.arange(n)-v[0][0]),4)*hn)/(np.power(v[0][1],2))-3,2)# kurtosis
    v[0][4] = round(-(hn[hn>0]*np.log(hn[hn>0])).sum(),2) # entropy
    v[0][5] = round(np.argmax(hn),2) # mode
    v[0][6:] = gt.hist_percentil(h,np.array([1,10,25,50,75,90,99])) #1,10,50,90,99% percentile

    return v
