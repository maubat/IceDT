import sys
sys.path.append('./libs')

import numpy as np
import geraIcebergsDB as gIDB
import getMetadata as gm
import getDescritores as gd
import ensemble as ens
import cv2
import histogram as hist
import myGeoTools as mgt
import matplotlib.pyplot as plt
import inspecao_frames as buscaice
#import os

from timeit import default_timer as timer
from scipy import ndimage
from myutils import imresize
from termcolor import colored
from IPython.display import clear_output # progressbar



def goClassification(segOutput,ices_df, img, committee, scene_info, show_plot_produto=False,
                     img_result=[], backfeed=False, fullplot=True, sub_giant=100):

    n_nibf = 0
    count_v = 0
    count_vp = 0

    nicebergs = 0
    nnonicebergs = 0

    print(colored('Classifying ', 'blue'), len(segOutput.idf), colored('objects.', 'green'))

    #s = timer()

    idf_validos = []
    morfo_track_1 = []

    cont_validos = 1
    cont_nicesbers = 1

    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 3)

    if fullplot:
        txts = 'Classifying small '
    else:
        txts = 'Classifying large '

    for i in segOutput.idf:

        clear_output(wait=True)
        bar_length = 50
        progress = float(i)/len(segOutput.idf)
        block = int(round(bar_length*progress))
        progressbar = "Classification Progress: [{0}] {1:.1f}% ({1:.1f})%".format("#"*block+"_"*(bar_length-block), progress*100.)
        print(colored(progressbar, "red")+colored(' detected icebergs: ', 'blue')+str(nicebergs))

        px_validos = np.asarray(segOutput.pixels_validos[i - 1]).astype(int)
        px_validos = ndimage.binary_fill_holes(px_validos)
        segOutput.pixels_validos[i - 1] = px_validos != 0

        xi, yi, xf, yf = segOutput.bbox[i - 1]

        pick_original = img[xi:xf, yi:yf]
        pick_original = pick_original * px_validos

        areNhole = sum(sum(w > 0 for w in pick_original))

        if sub_giant < 100:
            areNhole = (areNhole * (100.0 ** 2)) / sub_giant ** 2

        segOutput.areas[i - 1] = areNhole

        pick_original = cv2.bilateralFilter(pick_original, 3, 3, 3)
        pick_original = hist.adjust_gamma(pick_original, gamma=3.0)

        pick_original = imresize(pick_original, (32, 32), interp='bilinear')

        # plt.figure('pick')
        # plt.imshow(pick_original, cmap='gray')
        # plt.show()

        morpho_feats = []
        morpho_feats.append(segOutput.eccentricity[i - 1])
        morpho_feats.append(segOutput.eq_diameter[i - 1])
        morpho_feats.append(segOutput.solidity[i - 1])
        morpho_feats.append(segOutput.density_pixel[i - 1])
        morpho_feats.append(segOutput.convex_per[i - 1])
        morpho_feats.append(segOutput.per_index[i - 1])
        morpho_feats.append(segOutput.frac1[i - 1])
        morpho_feats.append(segOutput.frac2[i - 1])

        #Using original image signature
        feats_candidado = np.asarray(gd.getDescritores(pick_original, morpho_feats, hera=True))

        if backfeed:
            feats_candidado_base = np.asarray(gd.getDescritores(pick_original, morpho_feats, hera=True))

        feats_to_class = feats_candidado[:-1].reshape(1, -1)
        feats_to_class[np.isnan(feats_to_class)] = 0
        feats_to_class[np.isinf(feats_to_class)] = 0

        resul_ens = ens.doEnsemble(committee, feats_to_class, conf=0.51, trust_lvl=0.9, harmony_lvl=0.5)

        scp_ice = int(resul_ens[1][0, 0] * 100)
        scp_nice = int(resul_ens[1][0, 1] * 100)

        count_v += resul_ens[2]
        count_vp += resul_ens[3]

        if resul_ens[0] == 1:

            nicebergs += 1
            #print(colored("Icebergs: ", "blue"), nicebergs)

            idf_validos.append(True)
            morfo_track_1.append(buscaice.get_morfologia(segOutput.convex_pick[i - 1]))

            # Backfeed training database
            if backfeed:
                if not fullplot:
                    path_1 = '/xxx/xxx/xxxx'
                    if resul_ens[1][0, 0] > 0.51:
                        cv2.putText(pick_original, str(scp_ice), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2)
                        gIDB.backFeedingDB(pick_original, feats_candidado_base, path_1, scene_info['image_name'], nicebergs)
                        cont_validos += 1
                        cont_nicesbers += 1
        else:
            nnonicebergs += 1
            #print(colored("Icebergs: ", "blue"), nicebergs)

            idf_validos.append(False)
            morfo_track_1.append(0)

            if backfeed:

                if not fullplot:
                    path_2 = '/xx/xx/xx'
                    if resul_ens[1][0, 1] > 0.51:
                        if n_nibf % 1 == 0:
                            feats_candidado_base[-1] = 2.0
                            cv2.putText(pick_original, str(scp_nice), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 2)
                            gIDB.backFeedingDB(pick_original, feats_candidado_base, path_2, scene_info['image_name'],
                                               cont_nicesbers)
                            cont_nicesbers += 1

                    n_nibf += 1
    e = timer()

    # print("")
    # print(colored('Done! ', 'blue'), round((e - s) / 60, 3), colored(' min', 'blue'))
    # print(colored("Detected Icebegs: ", 'blue'), nicebergs)
    # print("Count majorityVote: ", count_v)
    # print("Count weighted votes: ", count_vp)

    ## To get and/or save iceberg Metadata ##################################################
    if nicebergs > 0:
        print(colored('Computing metadata ...', 'blue'))
        #os.remove(arquivo_path + 'resultados/' + f_name + '.txt')
        # Resizing data to improve performance it must be carefuly considered due small objects can be removed
        perc = sub_giant
        reshape_perc = perc / 100.0
        reshape_perc_ajuste_lins, reshape_perc_ajuste_cols = scene_info['rows'] % (100 / perc), \
                                                             scene_info['colums'] % (100 / perc)

        ices_df = gm.get_Metadata(scene_info['file_path'], scene_info['image_name'], scene_info['year'],
                        scene_info['date_img'], scene_info['ds'], segOutput.idf, idf_validos,
                        segOutput.pxcentro, segOutput.areas, segOutput.maior_eixo,
                        segOutput.menor_eixo, segOutput.perimetro, scene_info['pixel_size'],
                        scene_info['pixel_area'], reshape_perc, reshape_perc_ajuste_lins,
                        reshape_perc_ajuste_cols, morfo_track_1, ices_df, export=True)

    if img_result == []:
        img_result = np.zeros_like(img)

    img_result = mgt.make_plot(scene_info['file_path'], scene_info['image_name'], segOutput, img_result,
                               scene_info['land_pixels'], scene_info['area_pixels'], idf_validos,
                               scene_info['date_img'], scene_info['corners_wgs'], inset=True, plot_original=False,
                               show_plot=show_plot_produto, save_identify=True, showimgarea=True, showgrid=True,
                               showlabel=False, fullplot=fullplot)

    return nicebergs, img_result, ices_df
