import myGeoTools as mgt
import numpy as np
import sys

def get_iceberg_px2majorAxis(majoraxispx, pixel_size):

    majoraxispx = int(majoraxispx)-1
    ekm = (majoraxispx * pixel_size) * 1e-3
    ekm = round(ekm, 3)

    return ekm

def get_iceberg_px2area(areapx, perimetropx, pixel_size):

    areapx = int(areapx)
    perimetropx = int(perimetropx)
    
    pixel_area = pixel_size**2

    # akm = ((areapx - (perimetropx/2)) * pixel_area) * 1e-6
    akm = (areapx * pixel_area) * 1e-6
    akm = round(akm, 3)

    return akm

def get_iceberg_px2vol(areapx, perimetropx, pixel_size, keel=250):

    akm = get_iceberg_px2area(areapx, perimetropx, pixel_size)
    vol = (akm * 1e6) * keel # metro cubico
    vol = round(vol, 3)

    return vol

def get_iceberg_mass(areapx, perimetropx, pixel_size, keel=250):

    density = 850  # Stern et. al. 2016
    vol = get_iceberg_px2vol(areapx, perimetropx, pixel_size, keel)
    mass = (vol * density) * 1e-12 #Gt
    mass = round(mass, 3)

    return mass

def get_Metadata(arquivo_path,image_name, ano_img, data_img, ds, idf,idf_validos, pxcentro, areas, maior_eixo,
                 menor_eixo,perimetro, pixel_size, pixel_area, reshape_perc,reshape_perc_ajuste_lins,
                 reshape_perc_ajuste_cols,morfo_track_1, ices_df, export=False):

    # Convertendo dados em metros
    areaskm = []
    maioreixokm = []
    menoreixokm = []
    perimetrokm = []
    ice_class = []
    latcentro = []
    loncentro = []
    tracks_1 = []
    cont_validos = 0

    #To track
    maior_eixo_px = []
    menor_eixo_px = []
    area_px = []
    perimetro_px = []

    for i in idf:

        if idf_validos[i-1]:

            cont_validos += 1

            #Salvando morfologias para tracking dos icebergs validos
            tracks_1.append(morfo_track_1[i-1])
            maior_eixo_px.append(maior_eixo[i-1])
            menor_eixo_px.append(menor_eixo[i-1])
            area_px.append(areas[i-1])
            perimetro_px.append(perimetro[i-1])

            # cacluclo area (Km2)
            akm = get_iceberg_px2area(areas[i - 1], perimetro[i - 1], pixel_size)
            areaskm.append(akm)

            # Calculo maior eixo para classificar quanto ao tamanho (km)
            ekm = get_iceberg_px2majorAxis(maior_eixo[i - 1], pixel_size)
            maioreixokm.append(ekm)

            # Calculo menor eixo (km)
            emin = get_iceberg_px2majorAxis(menor_eixo[i - 1], pixel_size)
            menoreixokm.append(emin)

            # Perimetro (km)
            pkm = (perimetro[i - 1] * pixel_size) * 1e-3
            pkm = round(pkm, 3)
            perimetrokm.append(pkm)

            # Definindo classe pelo maior eixo (Wesche e Dierking, 2012)
            if ekm == 0:
                ice_class.append('Nodata')
            if ekm > 0 and ekm <= 5*1e-3:
                ice_class.append('Growler')
            if ekm > 5*1e-3 and ekm <= 15*1e-3:
                ice_class.append('BergyBit')
            if ekm > 15*1e-3 and ekm <= 60*1e-3:
                ice_class.append('Small')
            if ekm > 60*1e-3 and ekm <= 122*1e-3:
                ice_class.append('Medium')
            if ekm > 122*1e-3 and ekm <= 220*1e-3:
                ice_class.append('Large')
            if ekm > 220*1e-3 and ekm < 18000*1e-3:
                ice_class.append('VeryLarge')
            if ekm >= 18000*1e-3:
                ice_class.append('Giant')

            # Obtendo lat/lon centro
            ccc, lll = pxcentro[i - 1]

            lll = int(round((lll / reshape_perc) + reshape_perc_ajuste_lins))
            ccc = int(round((ccc / reshape_perc) + reshape_perc_ajuste_cols))

            lg, lt = mgt.pixel2coord(ds, ccc, lll)
            lg, lt = mgt.epsg3031toepsg4326(lg, lt)

            latcentro.append(lt)
            loncentro.append(lg)

    densidade = 850  # Stern et. al. 2016
    area_tot = 0
    vol_tot = 0
    vol_ices = []
    massa_fresh_ices = []
    for i in range(0, len(areaskm)):
        if ice_class[i] == 'BergyBit': keel = 2.5
        if ice_class[i] == 'Small': keel = 10
        if ice_class[i] == 'Medium': keel = 30
        if ice_class[i] == 'Large': keel = 60
        if ice_class[i] == 'VeryLarge': keel = 162
        if ice_class[i] == 'Giant': keel = 250

        volumeberg = round((areaskm[i] * (keel / 1e3)) * 1e9, 3)
        vol_ices.append(volumeberg)
        massa_fresh_ices.append(round((vol_ices[i] * densidade) * 1e-12, 3))

        area_tot += areaskm[i]
        vol_tot += volumeberg

    # area_tot = np.sum(areaskm)
    # vol_tot = (area_tot * (250 / 1e3)) * 1e9
    massa_fresh = (vol_tot * densidade) * 1e-12 # Em Gt

    for i in range(0, cont_validos):
        
        dataices = {'date': data_img,
                    'latitude': latcentro[i],
                    'longitude': loncentro[i],
                    'minoraxis_km': menoreixokm[i],
                    'majoraxis_km': maioreixokm[i],
                    'area_km2': areaskm[i],
                    'mass_Gt': massa_fresh_ices[i],
                    'perimeter_km': perimetrokm[i],
                    'sizeclass': ice_class[i]}
        
        ices_df = ices_df.append(dataices, ignore_index=True)
    
    return ices_df