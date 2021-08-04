import sys
sys.path.append('./libs')

import matplotlib.pyplot as plt
import imutils
import numpy as np
import geraIcebergsDB as gIDB
import getMetadata as gm

import getDescritores as gd
import getDescritores_2 as gd_2

import ensemble as ens
import cv2
import histogram as hist
import myGeoTools as mgt
import matplotlib.pyplot as plt
import inspecao_frames as buscaice

from skimage import exposure
from osgeo import gdal
from timeit import default_timer as timer
from skimage import img_as_ubyte
from scipy import ndimage, stats
from myutils import imresize, improve_scene
from termcolor import colored
from IPython.display import clear_output # progressbar

import os
baseDir = os.path.dirname(os.path.abspath('__file__')+'..')

def goClassification(segOutput,ices_df, img, committee, scene_info, show_plot_produto=False,
                     img_result=[], backfeed=False, fullplot=True, plot_original=False, save_identify=False, sub_giant=100):

    n_nibf = 0
    count_v = 0
    count_vp = 0

    nicebergs = 0
    nnonicebergs = 0

    print(colored('Classifying ', 'blue'), len(segOutput.idf), colored('objects.', 'green'))

    idf_validos = []
    morfo_track_1 = []

    cont_validos = 1
    cont_nicesbers = 1
    
    #if backfeed:
        #img_b = np.copy(img)  ###################################################
    
    if fullplot:
        txts = 'Classifying small '
        lbl = 'small'
        alg = 2
        deep_conf = 0.9

    else:
        txts = 'Classifying large '
        lbl = 'large'
        alg = 1
        deep_conf = 0.8
    
    img = improve_scene(img, alg=alg, classify=True)
        
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

        pick_original = img[xi:xf, yi:yf] * px_validos
        
        #if backfeed:
            #pick_b = img_b[xi:xf, yi:yf]
        #else:
            #pick_b = pick_original
            
        #pick_b = pick_b * px_validos
        
        areNhole = sum(sum(w > 0 for w in pick_original))

        if sub_giant < 100:
            areNhole = (areNhole * (100.0 ** 2)) / sub_giant ** 2

        segOutput.areas[i - 1] = areNhole
        
        pick_intensity = np.copy(pick_original)
        pick_intensity = imresize(pick_intensity, (32, 32), interp='bilinear').astype(float)
        pick_intensity[pick_intensity == 0.] = np.nan
        
        pick_avg = np.nanmean(pick_intensity)
        pick_var = round(np.sqrt(np.nanvar(pick_intensity)), 2)
        pick_mode = np.nanmax(stats.mode(pick_intensity))
    
        improve_pick = False
        h_i = 0.5
        if  99 < pick_avg < 210:
            gm1 = 1.7
            gm2 = 1.2
            improve_pick = True

        if  pick_avg <= 99:
            gm1 = 3.0
            gm2 = 1.5
            h_i = 0.4
            improve_pick = True
        
        if  improve_pick:
            pick_original = hist.adjust_gamma(pick_original, gamma=gm1)
            pick_original = exposure.adjust_sigmoid(pick_original)
            pick_original = hist.adjust_gamma(pick_original, gamma=gm2)
            
        pick_original = imresize(pick_original, (32, 32), interp='bilinear')
        
        pick_b = pick_original
        #pick_b = hist.adjust_gamma(pick_b, gamma=3.0)
        #pick_b = imresize(pick_b, (32, 32), interp='bilinear')
        '''
        if alg==1:
            print('AVG P: ' + str(pick_avg) + ' STD P: ' + str(pick_var) + 'MODE P: ' + str(pick_mode))
            areakmt = round((areNhole * (scene_info['pixel_size']**2)) * 1e-6, 3)
            print('Area: ', areakmt)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
            axes[0].imshow(hist.adjust_gamma(pick_b, gamma=3.0), cmap='gray')
            axes[1].imshow(pick_original, cmap='gray')
            plt.show()
        '''    
            
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
            feats_candidado_b = np.asarray(gd_2.getDescritores(pick_b, morpho_feats, hera=True))

        feats_to_class = feats_candidado[:-1].reshape(1, -1)
        feats_to_class[np.isnan(feats_to_class)] = 0
        feats_to_class[np.isinf(feats_to_class)] = 0

        resul_ens = ens.doEnsemble(committee, feats_to_class, conf=0.61, trust_lvl=0.9, harmony_lvl=h_i, deep_conf=deep_conf)

        scp_ice = int(resul_ens[1][0, 0] * 100)
        scp_nice = int(resul_ens[1][0, 1] * 100)

        count_v += resul_ens[2]
        count_vp += resul_ens[3]
        
        #print('RESULT: ', resul_ens[0], 'Ice Score: ', scp_ice, 'NO-Ice Score: ', scp_nice)
        area_tsh = round((areNhole * (scene_info['pixel_size']**2)) * 1e-6, 3)
        if resul_ens[0] == 1 and area_tsh < 5e3:

            nicebergs += 1

            idf_validos.append(True)
            morfo_track_1.append(buscaice.get_morfologia(segOutput.convex_pick[i - 1]))
            
            # Backfeed training database
            if backfeed:
                path_1 = baseDir+'/backfeed/iceberg/'
                #if resul_ens[1][0, 0] > 0.61:
                if alg == 1:
                    #cv2.putText(pick_b, str(scp_ice), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
                    gIDB.backFeedingDB(pick_b, feats_candidado_b, path_1, 'ice_'+lbl, scene_info['image_name'], nicebergs)
                    cont_validos += 1
        else:
            nnonicebergs += 1

            idf_validos.append(False)
            morfo_track_1.append(0)

            if backfeed:
                path_2 = baseDir+'/backfeed/noiceberg/'
                #if resul_ens[1][0, 1] > 0.61:
                if alg == 1:
                    if n_nibf % 1 == 0:
                        feats_candidado_b[-1] = 2.0
                        #cv2.putText(pick_b, str(scp_nice), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
                        gIDB.backFeedingDB(pick_b, feats_candidado_b, path_2, 'noice_'+lbl, scene_info['image_name'], cont_nicesbers)
                        cont_nicesbers += 1

                n_nibf += 1

    ## To get and/or save iceberg Metadata ##################################################
    if nicebergs > 0:
        print(colored('Computing metadata ...', 'blue'))
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
        
    if plot_original:
        img_base = mgt.make_plot(scene_info['file_path'], scene_info['image_name'], segOutput,
                                   img_as_ubyte(np.asarray(scene_info['ds'].GetRasterBand(1).ReadAsArray())),
                                   scene_info['land_pixels'], scene_info['area_pixels'], idf_validos,
                                   scene_info['date_img'], scene_info['corners_wgs'], inset=True, plot_original=True,
                                   show_plot=show_plot_produto, save_identify=save_identify, showimgarea=True, showgrid=True,
                                   showlabel=False, fullplot=fullplot)
        del img_base
        

    img_result = mgt.make_plot(scene_info['file_path'], scene_info['image_name'], segOutput, img_result,
                               scene_info['land_pixels'], scene_info['area_pixels'], idf_validos,
                               scene_info['date_img'], scene_info['corners_wgs'], inset=True, plot_original=False,
                               show_plot=show_plot_produto, save_identify=save_identify, showimgarea=True, showgrid=True,
                               showlabel=False, fullplot=fullplot)
    
    if save_identify:
        
        image_name = scene_info['image_name']
        if ".N1.gz.tif" in image_name:
            file_name, f_1, f_2, file_ext = image_name.split('.')

        elif ".N1.tif" in image_name:
            file_name, f_2, file_ext = image_name.split('.')

        elif ".tif" in image_name:
            file_name, file_ext = image_name.split('.')
                
        ices_df.to_csv(scene_info['file_path'] + 'resultados/' + file_name + '.txt', sep=' ', index=None)

    return nicebergs, img_result, ices_df
