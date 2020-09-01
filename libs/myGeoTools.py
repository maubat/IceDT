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

# %% Getting middle between two points
def get_middle(lat_base_ant, lon_base_ant, lat, lon):
    # Input values as degrees

    # Convert to radians
    lat1 = math.radians(lat_base_ant)
    lon1 = math.radians(lon_base_ant)
    lat2 = math.radians(lat)
    lon2 = math.radians(lon)

    bx = math.cos(lat2) * math.cos(lon2 - lon1)
    by = math.cos(lat2) * math.sin(lon2 - lon1)

    lat3 = math.atan2(math.sin(lat1) + math.sin(lat2),
                      math.sqrt((math.cos(lat1) + bx) * (math.cos(lat1) + bx) + by ** 2))
    lon3 = lon1 + math.atan2(by, math.cos(lat1) + bx)

    return math.degrees(lat3), math.degrees(lon3)


def make_track_plot_anima(path, filename, lat_s, lat_n, lon_w, lon_e, min_det=2, show_text=False, show_month=False, topo=False, imgid=0):

    print ('Ploting track . . .')

    num_lines = sum(1 for line in open(path))

    if num_lines >= min_det:

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.suptitle(filename)

        m = polar_stere(lon_w, lon_e, lat_s, lat_n)

        if not topo:
            m.drawmapboundary(fill_color='lightskyblue')

            path_shp = './AuxFiles/antarticamask'
            m.readshapefile(path_shp, 'antarticamask')

            patches = []
            for info, shape in zip(m.antarticamask_info, m.antarticamask):
                patches.append(Polygon(np.array(shape), True))

            ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))
        else:
            m.etopo()

        m.drawmeridians(np.arange(0, 360, 10), labels=[0, 1, 1, 0], linewidth=0.2)
        m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 1], linewidth=0.2)

        lats = []
        lons = []
        months = []
        speeds = []
        dates = []
        with open(path) as fb:

            for line in fb:

                vals = line.split(' ')

                data_ice_str = str(vals[0])
                # ano = int(data_ice_str[0:4])
                mes = int(data_ice_str[4:6])
                # dia = int(data_ice_str[6:8])

                # area = str(round(float(str(vals[4])), 2))

                lats.append(float(vals[1]))
                lons.append(float(vals[2]))
                months.append(mes)
                speeds.append(float(vals[5]))
                dates.append(vals[0])

        i = 0
        for l in range(0, len(lats)):

            if show_month:
                if 1 <= months[l] <= 6:
                # if (months[l] == 12) or (1 <= months[l] <= 4): Hartmut
                    cor = 'red'
                else:
                    cor = 'blue'

            if i > 0:
                x_ant, y_ant = x, y
                x, y = m(lons[l], lats[l])
            else:
                x, y = m(lons[l], lats[l])

            if i > 0:
                ax.annotate('', xy=(x, y), xycoords='data', xytext=(x_ant, y_ant),
                            textcoords='data', size=0.5,
                            arrowprops=dict(headwidth=10, \
                                            headlength=10, \
                                            facecolor=cor, \
                                            edgecolor="none", \
                                            connectionstyle="arc3, rad=0.001"))

            i+=1
            plt.title(dates[l],fontweight="bold")

            # site to make gif: https://ezgif.com/maker
            # plt.savefig("/home/csys/mbarbat/Schreibtisch/teste/Fig_"+ filename+ "_" + str(l) + ".png")   # To make a images to GIF

            plt.pause(0.0001)


        avgspeed = np.mean(speeds)
        avgspeed = round(avgspeed*86.4, 2)
        plt.text(x, y, "Avg Speed: "+str(avgspeed)+"Km/day")

        # plt.savefig("/home/csys/mbarbat/Schreibtisch/teste/Fig_"+ filename+ "_" + str(l+1) + ".png") # To make a images to GIF

        plt.pause(2)
        plt.show()

def anima_all(sea, save=False):

    if sea == "Weddell":
        lat_s = -77
        lat_n = -60
        lon_w = -65
        lon_e = -20
    else:
        lat_s = -78
        lat_n = -69
        lon_w = -200
        lon_e = -160

    start = 1

    path = '/scratch/users/mbarbat/Tracks_Processed/' + sea + '/P_No_Coast_velocities/'
    min_det = 2
    topo = False

    fig, ax = plt.subplots(figsize=(10, 8))
    m = polar_stere(lon_w, lon_e, lat_s, lat_n)


    if not topo:
        m.drawmapboundary(fill_color='lightskyblue')

        path_shp = './AuxFiles/antarticamask'
        m.readshapefile(path_shp, 'antarticamask')

        patches = []
        for info, shape in zip(m.antarticamask_info, m.antarticamask):
            patches.append(Polygon(np.array(shape), True))

        ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))
    else:
        m.etopo()

    m.drawmeridians(np.arange(0, 360, 10), labels=[0, 1, 1, 0], linewidth=0.2)
    m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 1], linewidth=0.2)

    for count, file_name in enumerate(sorted(os.listdir(path)), start=0):

        path = '/scratch/users/mbarbat/Tracks_Processed/' + sea + '/P_No_Coast_velocities/'
        filename = 'speed_Berg_' + str(start + count) + '.txt'
        path = path + filename

        imgid = count + 1

        print ("Ploting: ", imgid)

        num_lines = sum(1 for line in open(path))

        if num_lines >= min_det:

            lats = []
            lons = []
            months = []
            speeds = []
            dates = []
            with open(path) as fb:

                for line in fb:

                    vals = line.split(' ')

                    data_ice_str = str(vals[0])
                    # ano = int(data_ice_str[0:4])
                    mes = int(data_ice_str[4:6])
                    # dia = int(data_ice_str[6:8])

                    # area = str(round(float(str(vals[4])), 2))

                    lats.append(float(vals[1]))
                    lons.append(float(vals[2]))
                    months.append(mes)
                    speeds.append(float(vals[5]))
                    dates.append(vals[0])

            i = 0
            arrows = []
            for l in range(0, len(lats)):

                if i > 0:
                    x_ant, y_ant = x, y
                    x, y = m(lons[l], lats[l])
                else:
                    x, y = m(lons[l], lats[l])

                if i > 0:
                    an = ax.annotate('', xy=(x, y), xycoords='data', xytext=(x_ant, y_ant),
                                textcoords='data', size=0.5,
                                arrowprops=dict(headwidth=10, \
                                                headlength=10, \
                                                facecolor="red", \
                                                edgecolor="none", \
                                                connectionstyle="arc3, rad=0.001"))
                    arrows.append(an)

                i += 1

                plt.suptitle("Date: "+dates[l], fontweight="bold")

                if save:
                    # site to make gif: https://ezgif.com/maker
                    plt.savefig("/home/csys/mbarbat/Schreibtisch/teste/"+sea+"/Fig_"+ filename+ "_" + str(l) + ".png")   # To make a images to GIF

                plt.pause(0.0001)


            # Plot Avg Speed
            # avgspeed = np.mean(speeds)
            # avgspeed = round(avgspeed*86.4, 2)
            # plt.text(x, y, "Avg Speed: "+str(avgspeed)+"Km/day")

            if save:
                plt.savefig("/home/csys/mbarbat/Schreibtisch/teste/"+sea+"/Fig_"+ filename+ "_" + str(l+1) + ".png") # To make a images to GIF

            # plt.close()

            for ar in range(0, len(arrows)):
                arrows[ar].arrow_patch.set_color("blue")

            if save:
                plt.savefig("/home/csys/mbarbat/Schreibtisch/teste/" + sea + "/Fig_" + filename + "_" + str(l + 2) + ".png")  # To make a images to GIF

            # plt.pause(2)
            # plt.show()

def make_track_plot_unique(path, filename, lat_s, lat_n, lon_w, lon_e, show_text=False, show_month=False, topo=False):

    print ('Ploting track . . .')

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.suptitle(filename)

    m = polar_stere(lon_w, lon_e, lat_s, lat_n)

    if not topo:
        m.drawmapboundary(fill_color='lightskyblue')

        path_shp = './AuxFiles/antarticamask'
        m.readshapefile(path_shp, 'antarticamask')

        patches = []
        for info, shape in zip(m.antarticamask_info, m.antarticamask):
            patches.append(Polygon(np.array(shape), True))

        ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))
    else:
        m.etopo()

    m.drawmeridians(np.arange(0, 360, 10), labels=[0, 1, 1, 0], linewidth=0.2)
    m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 1], linewidth=0.2)

    cores = ['r', 'g', 'b', 'yellow', 'violet', 'pink', 'gray', 'brown', 'coral', 'lightgreen',
             'orange', 'magenta']

    cor_i = 0

    with open(path) as fb:

        i = 0

        for line in fb:

            vals = line.split(' ')


            xices = []
            yices = []

            c = 0
            x2 = 0
            y2 = 0
            coords2 = 0

            data = int(float(str(vals[0])))

            data_ice_str = str(vals[0])
            ano = int(data_ice_str[0:4])
            mes = int(data_ice_str[4:6])
            dia = int(data_ice_str[6:8])

            area = str(round(float(str(vals[4])), 2))

            if show_month:
                if 1 <= mes <= 6:
                    cor = cores[0]
                else:
                    cor = cores[2]

                # if mes == 1: cor = cores[0]
                # if mes == 2: cor = cores[1]
                # if mes == 3: cor = cores[2]
                # if mes == 4: cor = cores[3]
                # if mes == 5: cor = cores[4]
                # if mes == 6: cor = cores[5]
                # if mes == 7: cor = cores[6]
                # if mes == 8: cor = cores[7]
                # if mes == 9: cor = cores[8]
                # if mes == 10: cor = cores[9]
                # if mes == 11: cor = cores[10]
                # if mes == 12: cor = cores[11]

            if i > 0:
                x_ant, y_ant = x, y

                coords2 = coords1
                coords1 = (float(vals[1]), float(vals[2]))
                latm, lonm = get_middle(coords1[0],coords1[1], coords2[0],coords2[1])
                x2, y2 = m(lonm, latm)
                c = round(geodist.vincenty(coords1, coords2).km, 3)
                x, y = m(vals[2], vals[1])
            else:
                x, y = m(vals[2], vals[1])
                coords1 = (float(vals[1]), float(vals[2]))

            if i > 0:
                ax.annotate('', xy=(x, y), xycoords='data', xytext=(x_ant, y_ant),
                            textcoords='data', size=0.5,
                            arrowprops=dict(headwidth=10, \
                                            headlength=10, \
                                            facecolor=cor, \
                                            edgecolor="none", \
                                            connectionstyle="arc3, rad=0.001"))

            yices.append(y)
            xices.append(x)

            if show_text:
                plt.text(x, y, str(data) + ' (' + str(area) + 'km2)')
                plt.text(x2, y2, ' (' + str(c) + ' km)')

            m.plot(xices, yices, 'bo', markersize=8, linewidth=2, color='w')

            i+=1

    plt.show()
# %%
def make_track_plot_all_lines(path, lat_s, lat_n, lon_w, lon_e, min_det=5, show_text=False,show_month=False):

    print ('Ploting track . . .')

    fig, ax = plt.subplots(figsize=(10, 8))

    m = polar_stere(lon_w, lon_e, lat_s, lat_n)
    m.drawmapboundary(fill_color='lightskyblue')

    path_shp = './AuxFiles/antarticamask'
    m.readshapefile(path_shp, 'antarticamask')

    patches = []
    for info, shape in zip(m.antarticamask_info, m.antarticamask):
        patches.append(Polygon(np.array(shape), True))

    ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))

    m.drawmeridians(np.arange(0, 360, 10), labels=[0, 0, 0, 1], linewidth=0.2)
    m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 0], linewidth=0.2)

    cores = ['r', 'g', 'b', 'yellow', 'violet', 'pink', 'gray', 'brown', 'coral', 'lightgreen',
             'orange', 'magenta']

    cor_i = 0

    for file_name in sorted(os.listdir(path)):

        if file_name.endswith(".txt"):

            num_lines = sum(1 for line in open(path+file_name))

            if num_lines >= min_det:

                lats = []
                lons = []

                with open(path+file_name) as fb:

                    i = 0

                    for line in fb:

                        vals = line.split(' ')

                        data_ice_str = str(vals[0])
                        ano = int(data_ice_str[0:4])
                        mes = int(data_ice_str[4:6])
                        dia = int(data_ice_str[6:8])

                        area = str(round(float(str(vals[4])), 2))

                        if show_month:
                            if 1 <= mes <= 6:
                                cor = cores[0]
                            else:
                                cor = cores[2]

                        lats.append(float(vals[1]))
                        lons.append(float(vals[2]))
                        i+=1

                    xices,yices = m(lons,lats)
                    m.plot(xices, yices, '-', markersize=1, linewidth=4, color='r')
    plt.show()


def make_track_plot_all_initial_pos(path, lat_s, lat_n, lon_w, lon_e, min_det=5, show_text=False):

    print ('Ploting track . . .')

    # path_out = "/media/mauro/HD_ROSS/BACKUP SANDUICHE/Tracks_Processed/Weddell/Tracks_per_Size/"
    # path_out = "/media/mauro/HD_ROSS/BACKUP SANDUICHE/Tracks_Processed/Weddell/Tracks_per_Sector/"

    fig, ax = plt.subplots(figsize=(10, 8))

    m = polar_stere(lon_w, lon_e, lat_s, lat_n)
    m.drawmapboundary(fill_color='lightskyblue')

    path_shp = './AuxFiles/antarticamask'
    m.readshapefile(path_shp, 'antarticamask')

    patches = []
    for info, shape in zip(m.antarticamask_info, m.antarticamask):
        patches.append(Polygon(np.array(shape), True))

    ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))

    m.drawmeridians(np.arange(0, 360, 10), labels=[0, 0, 0, 1], linewidth=0.2)
    m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 0], linewidth=0.2)

    ices_cont = 0
    cor = "black"
    cont10 = 0
    cont100 = 0
    cont1000 = 0
    contm1000 = 0
    areas = []
    for file_name in sorted(os.listdir(path)):

        if file_name.endswith(".txt"):


            num_lines = sum(1 for line in open(path + file_name))

            if num_lines >= min_det:

                with open(path+file_name) as fb:

                    ices_cont += 1
                    # print ices_cont

                    for line in fb:

                        vals = line.split(' ')

                        # data = int(float(str(vals[0])))
                        area = float(str(vals[4]))
                        areas.append(area)

                        x, y = m(vals[2], vals[1])

                        # plt.text(x, y, str(ices_cont))

                        # if float(vals[1]) >= -72.0 and float(vals[2]) <= -50.0:
                        #     shutil.copy(path + file_name, path_out + "Larsen/" + file_name)
                        #
                        # if float(vals[1]) < -72.0 and float(vals[2]) < -28.0:
                        #     shutil.copy(path + file_name, path_out + "Ronne_Filchner/" + file_name)
                        #
                        # if float(vals[1]) < -70.0 and float(vals[2]) >= -25.0:
                        #     shutil.copy(path + file_name, path_out + "Riiser_Larsen/" + file_name)
                        #
                        # if (float(vals[1]) > -72.0 and float(vals[2]) > -50.0) and \
                        #         not(float(vals[1]) < -70.0 and float(vals[2]) >= -25.0):
                        #     shutil.copy(path + file_name, path_out + "Unknow/" + file_name)

                        # m.plot(xices, yices, '-', linewidth=2, color='r')

                        if area <= 10.0:
                            cor = "blue"
                            cont10 += 1
                            # shutil.copy(path + file_name, path_out + "1-10/" + file_name)

                        if 10.0 < area <= 100.0:
                            cor = "green"
                            cont100 += 1
                            # shutil.copy(path + file_name, path_out + "10-100/" + file_name)

                        if 100.0 < area <= 1000.0:
                            cor = "orange"
                            cont1000 += 1
                            # shutil.copy(path + file_name, path_out + "100-1000/" + file_name)

                        if area > 1000.0:
                            cor = "red"
                            contm1000 += 1
                            # shutil.copy(path + file_name, path_out + "1000-10000/" + file_name)

                        m.plot(x, y, 'bo', markersize=4, linewidth=2, color=cor)
                        break

    print ("A2: ", cont10)
    print ("A3: ", cont100)
    print ("A4: ", cont1000)
    print ("A5: ", contm1000)
    print ("Total: ", cont10 + cont100 + cont1000 + contm1000)
    print ("Areas min-max: ", min(areas), max(areas))
    plt.show()

def make_track_plot_all(path, lat_s, lat_n, lon_w, lon_e, min_det=5, show_text=False, showsi = False):

    print ('Ploting track . . .')

    fig, ax = plt.subplots(figsize=(10, 8))

    m = polar_stere(lon_w, lon_e, lat_s, lat_n)
    m.drawmapboundary(fill_color='lightskyblue')
    # m.etopo()

    path_shp = '../AuxFiles/antarticamask'
    m.readshapefile(path_shp, 'antarticamask')

    if showsi:
        for i in range(2002, 2013):
            ano = str(i)
            path_shp3 = '/home/mauro/Downloads/seaice extend/min-max/minSI/' + ano
            m.readshapefile(path_shp3, ano)

    patches = []
    for info, shape in zip(m.antarticamask_info, m.antarticamask):
        patches.append(Polygon(np.array(shape), True))

    ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))

    m.drawmeridians(np.arange(0, 360, 10), labels=[0, 0, 1, 0], linewidth=0.2)
    m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 0], linewidth=0.2)

    # cores = ['r', 'g', 'b', 'yellow', 'violet', 'pink', 'gray', 'brown', 'coral', 'lightgreen',
    #          'orange', 'peru', 'magenta', 'olive', 'indigo', 'm', 'silver', 'lime', 'salmon']

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cores = [v for v in colors.values()]

    cor_i = 0
    ices_cont = 0

    for file_name in sorted(os.listdir(path)):

        if file_name.endswith(".txt"):

            num_lines = sum(1 for line in open(path+file_name))

            line_mark = 0

            if num_lines >= min_det:
                print (file_name)
                # os.rename("/media/mauroleoc/HD_ROSS/BACKUP SANDUICHE/Tracks_Processed/Weddell/P_MASS/"+file_name,
                #           "/media/mauroleoc/HD_ROSS/BACKUP SANDUICHE/Tracks_Processed/Weddell/P_MASS/"+"A_"+file_name)

                with open(path+file_name) as fb:

                    if cor_i >= len(cores):
                        cor_i = 0

                    cor = cores[cor_i]
                    cor_i += 1

                    ices_cont += 1
                    # print ices_cont

                    i = 0

                    for line in fb:

                        vals = line.split(' ')


                        xices = []
                        yices = []

                        c = 0
                        x2 = 0
                        y2 = 0
                        coords2 = 0

                        data = int(float(str(vals[0])))
                        area = str(round(float(str(vals[4])), 2))

                        if i > 0:
                            x_ant, y_ant = x, y

                            coords2 = coords1
                            coords1 = (float(vals[1]), float(vals[2]))
                            latm, lonm = get_middle(coords1[0],coords1[1], coords2[0],coords2[1])
                            x2, y2 = m(lonm, latm)
                            c = round(geodist.vincenty(coords1, coords2).km, 3)
                            x, y = m(vals[2], vals[1])
                        else:
                            x, y = m(vals[2], vals[1])
                            coords1 = (float(vals[1]), float(vals[2]))

                        if i > 0:
                            ax.annotate('', xy=(x, y), xycoords='data', xytext=(x_ant, y_ant),
                                        textcoords='data', size=0.5,
                                        arrowprops=dict(headwidth=6, \
                                                        headlength=6, \
                                                        width=1.5,
                                                        facecolor=cor, \
                                                        edgecolor="none", \
                                                        connectionstyle="arc3, rad=0.001"))

                        yices.append(y)
                        xices.append(x)

                        # show_text = True
                        # if show_text:
                            # if line_mark % 1 == 0:
                            # plt.text(x, y, str(data) + ' (' + str(area) + 'km2)', fontsize=8)
                            # plt.text(x2, y2, ' (' + str(c) + ' km)', fontsize=8)
                            # plt.text(x, y, file_name, fontsize=8)

                        # m.plot(xices, yices, '-', linewidth=2, color='r')
                        # if area <= 300:
                        # m.plot(xices, yices, 'bo', markersize=8, linewidth=2, color='w')

                        line_mark += 1
                        i+=1

    plt.show()

def make_track_plot_all_TR(path, lat_s, lat_n, lon_w, lon_e, min_det=5, show_text=False, showsi = False):

    print ('Ploting track . . .')

    fig, ax = plt.subplots(figsize=(10, 8))

    m = polar_stere(lon_w, lon_e, lat_s, lat_n)
    m.drawmapboundary(fill_color='lightskyblue')
    # m.etopo()

    path_shp = './AuxFiles/antarticamask'
    m.readshapefile(path_shp, 'antarticamask')

    if showsi:
        for i in range(2002, 2013):
            ano = str(i)
            path_shp3 = '/home/mauro/Downloads/seaice extend/min-max/minSI/' + ano
            m.readshapefile(path_shp3, ano)

    patches = []
    for info, shape in zip(m.antarticamask_info, m.antarticamask):
        patches.append(Polygon(np.array(shape), True))

    ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))

    m.drawmeridians(np.arange(0, 360, 10), labels=[0, 0, 1, 0], linewidth=0.2)
    m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 0], linewidth=0.2)

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cores = [v for v in colors.values()]

    cor_i = 0
    ices_cont = 0

    for file_name in sorted(os.listdir(path)):

        if file_name.endswith(".txt"):

            num_lines = sum(1 for line in open(path+file_name))

            line_mark = 0

            if num_lines > min_det:
                print (file_name)

                with open(path+file_name) as fb:

                    if cor_i >= len(cores):
                        cor_i = 0

                    cor = cores[cor_i]
                    cor_i += 1

                    ices_cont += 1

                    i = 0
                    for line in fb:

                        vals = line.split(' ')

                        xices = []
                        yices = []

                        if i > 0:
                            x_ant, y_ant = x, y

                            x, y = m(vals[2], vals[1])
                        else:
                            x, y = m(vals[2], vals[1])

                        if i > 0:
                            ax.annotate('', xy=(x, y), xycoords='data', xytext=(x_ant, y_ant),
                                        textcoords='data', size=0.5,
                                        arrowprops=dict(headwidth=6, \
                                                        headlength=6, \
                                                        width=1.5,
                                                        facecolor=cor, \
                                                        edgecolor="none", \
                                                        connectionstyle="arc3, rad=0.001"))

                        yices.append(y)
                        xices.append(x)

                        line_mark += 1
                        i+=1

    plt.show()

def make_track_plot(path, lat_s, lat_n, lon_w, lon_e,clice,lstart,lend, save_plot=False, show_text=False):

    print ('Ploting track . . .')

    path_file = path + 'tracks_Filtered_'+clice+'_'+lstart+'_'+lend+'.txt'
    path_aux = path + 'tracks_aux_Filtered_'+clice+'_'+lstart+'_'+lend+'.txt'

    if os.path.isfile(path_file):

        fig, ax = plt.subplots(figsize=(10, 8))

        m = polar_stere(lon_w, lon_e, lat_s, lat_n)
        m.drawmapboundary(fill_color='lightskyblue')

        path_shp = './AuxFiles/antarticamask'
        m.readshapefile(path_shp, 'antarticamask')

        patches = []
        for info, shape in zip(m.antarticamask_info, m.antarticamask):
            patches.append(Polygon(np.array(shape), True))

        ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))

        m.drawmeridians(np.arange(0, 360, 10), labels=[0, 1, 1, 0], linewidth=0.2)
        m.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 1], linewidth=0.2)

        # m.drawmapscale(-75, -77, -45, -75, 300, barstyle='fancy')

        # Leitura dos arquivos para plot
        # Desenha raio
        #        equi(m, lon_base, lat_base, radius_lim, lw=2)
        cores = ['r', 'g', 'b', 'yellow', 'violet', 'pink', 'gray', 'brown', 'coral', 'lightgreen',
                 'orange', 'peru', 'magenta', 'olive', 'indigo', 'm', 'silver', 'lime', 'salmon']

        cor_i = 0
        ices = 0
        with open(path_file) as fb:

            for line in fb:

                ices += 1

                if cor_i > 18:
                    cor_i = 0

                cor = cores[cor_i]
                cor_i += 1

                vals = line.split(' ')

                n_vals = (len(vals) - 1)

                xices = []
                yices = []

                cont = 0

                for i in range(0, n_vals, 5):

                    cont += 1
                    data = int(float(str(vals[i])))
                    area = str(round(float(str(vals[i + 4])), 2))

                    if i > 0:
                        x_ant, y_ant = m(vals[j + 2], vals[j + 1])

                    x, y = m(vals[i + 2], vals[i + 1])

                    if i > 0:
                        ax.annotate('', xy=(x, y), xycoords='data', xytext=(x_ant, y_ant),
                                    textcoords='data', size=1.5,
                                    arrowprops=dict(headwidth=12, \
                                                    headlength=12, \
                                                    facecolor=cor, \
                                                    edgecolor="none", \
                                                    connectionstyle="arc3, rad=0.001"))
                    j = i

                    yices.append(y)
                    xices.append(x)

                    if show_text:
                        plt.text(x, y, str(data) + ' (' + str(area) + 'km2)')

                # m.plot(xices, yices, '-', linewidth=1, color='r')
                m.plot(xices, yices, 'bo', markersize=8, linewidth=3, color='w')

        with open(path_aux) as fb:

            for line in fb:

                vals = line.split(' ')

                n_vals = (len(vals) - 1)

                xm = []
                ym = []

                for i in range(0, n_vals, 3):
                    x, y = m(vals[i + 1], vals[i])

                    ym.append(y)
                    xm.append(x)

                    if show_text:
                        plt.text(x, y, ' (' + str(vals[i + 2]) + ' km)')

                # m.plot(xm, ym, 'bo', markersize=6, color='green')

        # plt.tight_layout()
        print ('Total of icebergs tracked: ', ices)
        if save_plot:
            file_name = 'Tracks_'+clice+'_'+lstart+'_'+lend
            plt.savefig(path+'plots/' + file_name+'.tif', dpi=300, format='tif')
        else:
            plt.show()

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

        # img_result = np.zeros_like(img_result)
        # img_result[area_pixels] = 75
        # img_result[land_pixels[0], land_pixels[1]] = 110

        for i in idf:
            if idf_validos[i - 1]:
                cd = cds[i - 1]
                img_result[cd[:, 0], cd[:, 1]] = 220

        #img_result[land_pixels[0], land_pixels[1]] = land_pixels[2]
        #img_result = cv2.cvtColor(img_result, cv2.COLOR_GRAY2RGB)

        # for i in idf:
        #     if idf_validos[i - 1]:
        #         cd = cds[i - 1]
        #         img_result[cd[:, 0], cd[:, 1], 0] = 0
        #         img_result[cd[:, 0], cd[:, 1], 1] = 148
        #         img_result[cd[:, 0], cd[:, 1], 2] = 255

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

        # Desenhar meridianos e paralelos.
        # labels = [left,right,top,bottom]
        if showgrid:
            # dashes=[tam pixel, numero de pulos]
            m.drawparallels(np.arange(-90., 90., 2.), labels=[False, False, False, True], color='w', linewidth=0.5)
            m.drawmeridians(np.arange(-180., 180., 3.), labels=[True, False, False, False], color='w', linewidth=0.5)

        else:
            m.drawparallels(np.arange(-90., 90., 2.), labels=[False, False, False, True], color='w', linewidth=0.01)
            m.drawmeridians(np.arange(-180., 180., 3.), labels=[True, False, False, False], color='w', linewidth=0.01)

        m.ax = ax

        if inset:
            axin = inset_axes(m.ax, width="20%", height="20%", loc=2)
            # Global inset map.
            inmap = Basemap(projection='spstere', boundinglat=-62, lat_0=-90, lat_ts=-71,
                            lon_0=-180, resolution='l', ax=axin, anchor='NE')

            inmap.drawcoastlines()
            inmap.fillcontinents(color='b')

            lats = [ullat, lllat, lrlat, urlat]
            lons = [ullon, lllon, lrlon, urlon]

            draw_screen_poly(lats, lons, inmap)

        print(labl)
        plt.show()
        #if show_plot: plt.show()

        #if save_identify:

            #file_name, file_ext = image_name.split('.')

            #if plot_original: file_name = file_name+'_base'
            #plt.savefig(arquivo_path + 'resultados/' + file_name + '.png', bbox_inches='tight', dpi=300, format='png')
            #plt.clf()

    return img_result