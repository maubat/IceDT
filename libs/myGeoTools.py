import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import geopy.distance as geodist
import math

from mpl_toolkits.basemap import Basemap
from pyproj import Proj, transform
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import PatchCollection
from matplotlib import colors as mcolors
from scipy import ndimage
from skimage.filters import rank
from skimage.morphology import disk

# parametros da reprojecao
# Geodetic Datum : World Geodetic System 1984 (WGS84)
# Projection: South Polar Stereographic 3031
# Projection Parameter: Mudar Latitude of Origin para -90
# x - easting e y - northing
# function to generate .prj file information using spatialreference.org
# plt.rcParams.update({'font.size': 16})

def getWKT_PRJ (epsg_code):
     import urllib
     # access projection information
     wkt = urllib.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code))
     # remove spaces between charachters
     remove_spaces = wkt.read().replace(" ","")
     # place all the text on one line
     output = remove_spaces.replace("\n", "")
     return output

def criaWKTFiles():
    # create the .prj file
    epsg_code = '3031'
    prj = open("./projFiles/"+epsg_code+".prj", "w")
    # call the function and supply the epsg code
    epsg = getWKT_PRJ(epsg_code)
    prj.write(epsg)
    prj.close()

def epsg3031toepsg4326(x,y):

    # Referencias espaciais definidas explicitamente
    # http://spatialreference.org/ref/epsg/wgs-84-antarctic-polar-stereographic/
    # inProj = pyproj.Proj("+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    # http://spatialreference.org/ref/epsg/4326/
    # outProj = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ")

    inProj = Proj(init='epsg:3031')
    outProj = Proj(init='epsg:4326')

    return transform(inProj, outProj, x, y)

def pixel2coord(data, x, y):
    # print ds.GetMetadata()
    # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner,
    # a/e are pixel wight/height and b/d is rotation and is zero if image is north up.
    xoff, a, b, yoff, d, e = data.GetGeoTransform()
    xp = a * x + b * y + a * 0.5 + b * 0.5 + xoff
    yp = d * x + e * y + d * 0.5 + e * 0.5 + yoff
    # xp = a * x + b * y + xoff
    # yp = d * x + e * y + yoff

    return (xp, yp)


def coord2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    xx = int((x - ulX) / xDist)
    yy = int((y - ulY) / yDist)
    return (xx, yy)

def crop_image(geotrans, image , lonmin, latmin, lonmax, latmax):
    geoTrans = geotrans
    xmax, ymax = coord2Pixel(geoTrans, lonmax, latmax)
    xmin, ymin = coord2Pixel(geoTrans, lonmin, latmin)
    img_crop = image[ymin:ymax, xmin:xmax]
    return img_crop

def draw_screen_poly(lats, lons, m):
    x, y = m(lons, lats)
    xy = list(zip(x, y))
    poly = Polygon(xy, color='r', alpha = 0.8)
    plt.gca().add_patch(poly)

def polar_stere(lon_w, lon_e, lat_s, lat_n, **kwargs):
    '''Returns a Basemap object (NPS/SPS) focused in a region.
    lon_w, lon_e, lat_s, lat_n -- Graphic limits in geographical coordinates.
                                  W and S directions are negative.
    **kwargs -- Aditional arguments for Basemap object.
    '''

    # lat_0=-75, lat_ts=-71, lon_0=-45,

    lon_0 = lon_w + (lon_e - lon_w) / 2.
    ref = lat_s if abs(lat_s) > abs(lat_n) else lat_n
    lat_0 = np.copysign(90., ref)
    proj = 'npstere' if lat_0 > 0 else 'spstere'
    prj = Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0,
                          boundinglat=0, resolution='c')
    lons = [lon_w, lon_e, lon_w, lon_e, lon_0, lon_0]
    lats = [lat_s, lat_s, lat_n, lat_n, lat_s, lat_n]
    x, y = prj(lons, lats)
    ll_lon, ll_lat = prj(min(x), min(y), inverse=True)
    ur_lon, ur_lat = prj(max(x), max(y), inverse=True)
    return Basemap(projection='stere', lat_0=lat_0, lon_0=lon_0,
                           llcrnrlon=ll_lon, llcrnrlat=ll_lat,
                           urcrnrlon=ur_lon, urcrnrlat=ur_lat, **kwargs)


def make_plot(arquivo_path, image_name, segOutput, img_result, land_pixels, area_pixels, idf_validos, data_img,
              corners_wgs84, inset=True, plot_original=False, show_plot=False, save_identify=False, showimgarea=True,
              showgrid=True, showlabel=False, fullplot=True):

    pxcentro = segOutput.pxcentro
    idf = segOutput.idf
    cds = segOutput.coords

    # ullon, ullat, lllon, lllat, urlon, urlat, lrlon, lrlat
    ullon = corners_wgs84['UL'][0]
    ullat = corners_wgs84['UL'][1]
    lllon = corners_wgs84['LL'][0]
    lllat = corners_wgs84['LL'][1]
    urlon = corners_wgs84['UR'][0]
    urlat = corners_wgs84['UR'][1]
    lrlon = corners_wgs84['LR'][0]
    lrlat = corners_wgs84['LR'][1]

    #Plotar a imagem original para comparacoes
    if plot_original:

        img_result[land_pixels[0], land_pixels[1]] = land_pixels[2]
        
        labl = 'SAR Base to Classification.'

    else:
        
        labl = 'Classification product.'

        for i in idf:
            if idf_validos[i - 1]:
                cd = cds[i - 1]
                img_result[cd[:, 0], cd[:, 1]] = 220

    # Inserindo metadados sobre a imagem
    if fullplot:
        if showlabel:
            N_label = 1
            for i in idf:
                if idf_validos[i-1]:
                    cv2.putText(img_result, str(N_label), (pxcentro[i - 1][0], pxcentro[i - 1][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                    N_label += 1

        # Plotando Usando Basemap
        plt.clf()
        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 8))

        # plt.figure('Iceberg Tracker', figsize=(10, 8))
        plt.title('Identification Step ' + data_img)

        m = Basemap(llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat, epsg=3031)
        m.imshow(np.flipud(img_result), cmap='gray')
        plt.xlabel('Latitude', fontsize=12, labelpad=20)
        plt.ylabel('Longitude', fontsize=12, labelpad=40)

        if showgrid:
            m.drawparallels(np.arange(-90., 90., 2.), labels=[False, False, False, True], color='w', linewidth=0.5)
            m.drawmeridians(np.arange(-180., 180., 3.), labels=[True, False, False, False], color='w', linewidth=0.5)

        else:
            m.drawparallels(np.arange(-90., 90., 2.), labels=[False, False, False, True], color='w', linewidth=0.01)
            m.drawmeridians(np.arange(-180., 180., 3.), labels=[True, False, False, False], color='w', linewidth=0.01)

        m.ax = ax

        if inset:
            axin = inset_axes(m.ax, width="20%", height="20%", loc=2)
            inmap = Basemap(projection='spstere', boundinglat=-62, lat_0=-90, lat_ts=-71,
                            lon_0=-180, resolution='l', ax=axin, anchor='NE')

            inmap.drawcoastlines()
            inmap.fillcontinents(color='b')

            lats = [ullat, lllat, lrlat, urlat]
            lons = [ullon, lllon, lrlon, urlon]

            draw_screen_poly(lats, lons, inmap)

        
        if save_identify:

            #file_name, file_ext = image_name.split('.')
            if ".N1.gz.tif" in image_name:
                file_name, f_1, f_2, file_ext = image_name.split('.')

            elif ".N1.tif" in image_name:
                file_name, f_2, file_ext = image_name.split('.')

            elif ".tif" in image_name:
                file_name, file_ext = image_name.split('.')

            if plot_original: file_name = file_name+'_base'
            plt.savefig(arquivo_path + 'resultados/' + file_name + '.png', bbox_inches='tight', dpi=300, format='png')
            
            print(labl)
            plt.show()
            
        else:
            print(labl)
            plt.show()

    return img_result