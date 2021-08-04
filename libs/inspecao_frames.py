import sys
sys.path.append('./libs')

import numpy as np
import math
import fractal
import gc
import Structs as segData
import matplotlib.pyplot as plt
import segmentation as seg
import pandas as pd
import PIL
import histogram as hist
import cv2
import imutils

from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import rank
from skimage.morphology import disk
from skimage import img_as_float
from timeit import default_timer as timer
from myutils import imresize, convert, match_hist, improve_scene

def get_morfologia(pick):

    pick = cv2.threshold(pick, 127, 255, cv2.THRESH_BINARY)[1]
    l, c = pick.shape

    cm = ndimage.measurements.center_of_mass(pick)
    x_centro = int(cm[0])
    y_centro = int(cm[1])

    morfo_all = []

    for i in range(0, 360):
        d = 0
        x = x_centro
        y = y_centro

        while (pick[x, y] == 255 or pick[x, y] == 1) and x < l - 1 and x >= 0 and y < c - 1 and y >= 0:

            x = int(x_centro + d * np.cos(np.deg2rad(i)))
            y = int(y_centro + d * np.sin(np.deg2rad(i)))

            if x < 0:
                x = x_centro - x

            if y < 0:
                y = y_centro - y

            d += 1

        d -= 1

        morfo_all.append(d)
 
    return morfo_all

def buscaIce(templateg, rsp=100, alg=1, minseg=10, maxseg=np.inf):

    if rsp < 100:
        reshape_perc = rsp / 100.0
        templateg = imresize(templateg, reshape_perc, interp='bilinear')

    frames_usr = 4
    frames = 1

    xlin, xcol = np.where(rank.median(convert(templateg, 0, 255, np.uint8), disk(4)) > 5)
    if len(xlin) > 0 and len(xcol) > 0:
        minyb = np.min(xlin)
        maxyb = np.max(xlin)
        minxb = np.min(xcol)
        maxxb = np.max(xcol)

        sub_templateg_base = templateg[minyb:maxyb, minxb:maxxb]
    else:
        sub_templateg_base = np.copy(templateg)
        minyb = 0
        minxb = 0

    del templateg

    ll, cc = sub_templateg_base.shape

    if ll > 500000 and cc > 500000:
        frames = np.sqrt(frames_usr)
        print("Splited to improve performance in ", frames_usr, " frames.")

    if ll % 2 != 0: ll -= 1
    if cc % 2 != 0: cc -= 1

    stepll = int(ll / frames)
    stepcc = int(cc / frames)

    segOutput = segData.segmentData()
    ice_id = 1

    contFrame = 0
    for ill in range(0, int(stepll * frames), stepll):

        miny = ill
        maxy = ill + stepll

        for jll in range(0, int(stepcc * frames), stepcc):

            contFrame += 1
            #print('frame: ', contFrame)

            minx = jll
            maxx = jll + stepcc

            # preprocing to segmentation
            sub_templateg = sub_templateg_base[miny:maxy, minx:maxx]
                    
            ajuste_minx = minxb + minx
            ajuste_miny = minyb + miny

            if alg == 1:
                sub_templateg = cv2.fastNlMeansDenoising(sub_templateg, None, 10, 7, 21)
                sub_templateg = improve_scene(sub_templateg, alg=alg, classify=False)
                segmentos = seg.segFelzenszwalb(sub_templateg, 12, 30, 0, 3) # 19 30 0 3

            if alg == 2:
                sub_templateg = improve_scene(sub_templateg, alg=alg, classify=False)
                segmentos , coords = seg.region_based_segmentation(sub_templateg)

            gc.collect()

            del sub_templateg
            sub_templateg = sub_templateg_base[miny:maxy, minx:maxx]

            if rsp == 100:
                dsplim = 30
            else:
                dsplim = 10

            print('Computing segmentation metadata...')
            regions = regionprops(segmentos)

            for segmentoAtual in regions:

                ar_seg = segmentoAtual['area']

                if rsp < 100:
                    ar_seg = (ar_seg * (100.0**2)) / rsp**2

                if ar_seg >= minseg and ar_seg < maxseg:

                    per_seg = segmentoAtual['perimeter']

                    if rsp < 100:
                        per_seg = (per_seg * 100.0) / rsp

                    if per_seg > 0:

                        bbt = segmentoAtual['bbox']
                        pv = segmentoAtual['image']
                        xi, yi, xf, yf = bbt
                        pick1 = sub_templateg[xi:xf, yi:yf] * pv
                        pickback = sub_templateg[xi:xf, yi:yf]

                        pick1 = \
                            cv2.bilateralFilter(imresize(pick1, (32, 32), interp='bilinear'), 3, 3, 3).astype('float')
                        pickback = \
                            cv2.bilateralFilter(imresize(pickback, (32, 32), interp='bilinear'), 3, 3, 3).astype('float')

                        pickback[pick1 != 0] = np.NAN
                        pick1[pick1 == 0] = np.NAN

                        avgpick = np.nanmean(pick1)
                        avgpickback = np.nanmean(pickback)

                        dstpb = np.abs(avgpickback - avgpick)

                        if avgpick > 30 and dstpb > dsplim:  # 30 20

                            pick1 = sub_templateg[xi:xf, yi:yf] * pv
                            pick1 = imresize(pick1, (32, 32), interp='bilinear')

                            eixo_maior = int(round(segmentoAtual['major_axis_length']))
                            eixo_menor = int(round(segmentoAtual['minor_axis_length']))
                            
                            if rsp < 100:
                                eixo_maior = (eixo_maior * 100.0) / rsp
                                eixo_menor = (eixo_menor * 100.0) / rsp
                            
                            extent = segmentoAtual['extent']

                            normPeriIndex = (2 * math.sqrt(math.pi * ar_seg)) / per_seg

                            cd = segmentoAtual['coords']
                            for i in range(0, len(cd)):
                                cd[i][0] += ajuste_miny
                                cd[i][1] += ajuste_minx

                            centroid = segmentoAtual['centroid']
                            segOutput.areas.append(ar_seg)

                            [xi, yi, xf, yf] = segmentoAtual['bbox']
                            [xi, yi, xf, yf] = [xi + ajuste_miny, yi + ajuste_minx,
                                                xf + ajuste_miny, yf + ajuste_minx]

                            segOutput.bbox.append([xi, yi, xf, yf])
                            segOutput.maior_eixo.append(eixo_maior)
                            segOutput.menor_eixo.append(eixo_menor)
                            segOutput.perimetro.append(per_seg)

                            segOutput.coords.append(cd)

                            # Features morfologicas
                            segOutput.eccentricity.append(segmentoAtual['eccentricity'])
                            segOutput.eq_diameter.append(segmentoAtual['equivalent_diameter'])
                            segOutput.solidity.append(segmentoAtual['solidity'])
                            segOutput.density_pixel.append(extent)

                            polsby = ((4 * math.pi) * ar_seg) / per_seg ** 2
                            frac = fractal.slope_finder(pick1)

                            segOutput.convex_per.append(polsby)
                            segOutput.per_index.append(normPeriIndex)

                            segOutput.frac1.append(frac[2])
                            segOutput.frac2.append(frac[3])

                            segOutput.pixels_validos.append(segmentoAtual['image'])

                            a = int(round(centroid[1])) + ajuste_minx  # col
                            b = int(round(centroid[0])) + ajuste_miny  # row
                            c = (a, b)

                            segOutput.pxcentro.append(c)

                            segOutput.idf.append(ice_id)
                            ice_id += 1

                            pick2morpho = segmentoAtual['image'].astype('uint8')
                            pick2morpho[pick2morpho != 0] = 255

                            if rsp < 100:
                                lp, cp = pick2morpho.shape
                                lp = int((lp * 100.0) / rsp)
                                cp = int((cp * 100.0) / rsp)
                                pick2morpho = imresize(pick2morpho, (lp, cp), interp='bilinear')
                            
                            segOutput.convex_pick.append(pick2morpho)
                            
    return segOutput
