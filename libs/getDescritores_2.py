import sys
import numpy as np
import histogram as hist
import hist2stats as hist2st
import img2stats_2 as h2s
import gabor_filter as gbf
import matplotlib.pyplot as plt
import coocorrenciagray as ct


def getDescritores(img, morpho_feats, hera=False):

    feats = []

    #Calculando estatisticas de pixel
    img_feats = h2s.get_all_stats(img)
    # print img_feats

    feats = np.append(feats, img_feats[:,:4])
    # feats = np.append(feats, img_feats)

    #Calculando Distribuicao de frequencia de niveis de cinza (Histograma)
    h = hist.histogram(img)
    h[0] = 0 #Eliminando cor preta das metricas (preto ausencia de dados)

    #Calculando a tendencia linear no histograma para pegar o coef angular
    coef_ang, coef_lin = np.polyfit(np.arange(len(h)), h, 1)
    # coef_ang = round(coef_ang,2)
    # coef_lin = round(coef_lin,2)
    #tend = coef_ang*(np.arange(256)) + coef_lin
    # print "Coef angular: ", coef_ang, tend

    #Calculando estatisticas de distribuicao de frequencia
    hist_feats = hist2st.histogram2stats(h)
    # print hist_feats

    feats = np.append(feats, hist_feats[:,:6])
    # feats = np.append(feats, hist_feats)

    #Medidas de assimetria (Coeficientes de assimetria de Pearson)
    # CP_1 = (hist_feats[0, 0] - hist_feats[0, 5])/ np.sqrt(hist_feats[0,1]) #CP_1 = (media - moda)/desvio
    # CP_2 = 3 * (hist_feats[0, 0] - hist_feats[0, 9]) / np.sqrt(hist_feats[0, 1])  # CP_2 = 3*(media - mediana)/desvio
    # # CP_1 = 0 simetria
    # # CP_1 < 0 Assimetria a esquerda (Negativa)
    # # CP_1 > 0 Assimetria a direita  (Positiva)
    # if CP_1 == 0 or CP_2 == 0: assimetria = 0
    # if CP_1 < 0 or CP_2 < 0: assimetria = -1
    # if CP_1 > 0 or CP_2 > 0: assimetria = 1

    feats = np.append(feats, coef_ang)

    #Gabor Filter descritor textura frequencia
    gabor_feats = gbf.gabor_filter(img)
    # print 'Gabor Feats: ', gabor_feats

    feats = np.append(feats, gabor_feats)

    if hera:
        #Matriz de Haralick
        feats = np.append(feats, ct.coocorrencia(img, 1, 0))
        feats = np.append(feats, ct.coocorrencia(img, 1, np.pi/4))
        feats = np.append(feats, ct.coocorrencia(img, 1, np.pi/2))
        feats = np.append(feats, ct.coocorrencia(img, 1, (3 * np.pi)/4))

    #feats Morfologicas
    feats = np.append(feats, morpho_feats[0])
    feats = np.append(feats, morpho_feats[1])
    feats = np.append(feats, morpho_feats[2])
    feats = np.append(feats, morpho_feats[3])
    feats = np.append(feats, morpho_feats[4])
    feats = np.append(feats, morpho_feats[5])
    feats = np.append(feats, morpho_feats[6])
    feats = np.append(feats, morpho_feats[7])

    #identificador de Iceberg 1 para usar no treinamento
    feats = np.append(feats, 1)

    return feats