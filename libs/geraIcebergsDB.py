import cv2
import numpy as np
import getDescritores as gd
#from scipy.misc import imresize
from PIL import Image as imresize
import os.path

import os
baseDir = os.path.dirname(os.path.abspath('__file__')+'..')

def geraIceDB(path, img, image_name, segOutput):

    print('Gerando Icebergs DB')

    idf = segOutput.idf
    bbox = segOutput.bbox
    pixels_validos = segOutput.pixels_validos

    train_feats = []
    cont_picks = 1

    file_name, file_ext = image_name.split('.')

    for i in idf:

        xi, yi, xf, yf = bbox[i - 1]
        pick1 = img[xi:xf, yi:yf]
        pick1 = pick1 * pixels_validos[i - 1]
        pick1 = imresize(pick1, (32, 32), interp='bilinear')

        #Feats Morfologicas
        morpho_feats = []
        morpho_feats.append(segOutput.eccentricity[i - 1])
        morpho_feats.append(segOutput.eq_diameter[i - 1])
        morpho_feats.append(segOutput.solidity[i - 1])
        morpho_feats.append(segOutput.density_pixel[i - 1])
        morpho_feats.append(segOutput.convex_per[i - 1])
        morpho_feats.append(segOutput.per_index[i - 1])
        morpho_feats.append(segOutput.frac1[i - 1])
        morpho_feats.append(segOutput.frac2[i - 1])

        feats_temp = np.asarray(gd.getDescritores(pick1, morpho_feats, hera=True))
        train_feats.append(feats_temp)

        # salvando Picks Ice
        # Verificar se o nome ja existe para nao sobrescrever

        nomefig = 'ice_'+ file_name + '_' + str(cont_picks) + '.png'
        nometxt = 'ice_' + file_name + '_' + str(cont_picks) + '.txt'

        if not os.path.isfile(path+nomefig):
            #Salvando a png da amostra de ice
            cv2.imwrite(path + nomefig, pick1)
            # Salvando Features em arquivo
            file = open(path + nometxt, 'w')
            col_feats = len(feats_temp)
            for i in range(0, col_feats):
                file.write(str(train_feats[i]) + ' ')
            file.close()

        cont_picks += 1
        del pick1

def backFeedingDB(pick1, feats, path, lbl, image_name, cont_picks):
    # salvando Picks Ice
    # Verificar se o nome ja existe para nao sobrescrever

    if image_name.endswith(".N1.gz.tif"):
        file_name, f_1, f_2, file_ext = image_name.split('.')

    elif image_name.endswith(".N1.tif"):
        file_name, f_2, file_ext = image_name.split('.')

    else:
        file_name, file_ext = image_name.split('.')

    # file_name, file_ext = image_name.split('.')

    nomefig = lbl + '_' + file_name + '_' + str(cont_picks) + '.png'
    #nometxt = lbl + '_' + file_name + '_' + str(cont_picks) + '.txt'
    nometxt = baseDir+'/backfeed/training_dataset.txt'

    if not os.path.isfile(path + nomefig):
        # Salvando a png da amostra de ice
        cv2.imwrite(path + nomefig, pick1)
        # Salvando Features em arquivo
        #file = open(path + nometxt, 'w')
        file = open(nometxt, 'a')
        col_feats = len(feats)
        file.write(nomefig + ' ')
        for i in range(0, col_feats):
            file.write(str(feats[i]) + ' ')
        file.write('\n')
        file.close()

    del pick1
