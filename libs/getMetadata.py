import myGeoTools as mgt
import numpy as np
import sys

def get_Metadata(arquivo_path,image_name, ano_img, data_img, ds, idf,idf_validos, pxcentro, areas, maior_eixo,
                 menor_eixo,perimetro, pixel_size, pixel_area, reshape_perc,reshape_perc_ajuste_lins,
                 reshape_perc_ajuste_cols,morfo_track_1, ices_df, export=False):

    # Convertendo dados em metros
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
            
            # Obtendo lat/lon centro
            ccc, lll = pxcentro[i - 1]

            lll = int(round((lll / reshape_perc) + reshape_perc_ajuste_lins))
            ccc = int(round((ccc / reshape_perc) + reshape_perc_ajuste_cols))

            lg, lt = mgt.pixel2coord(ds, ccc, lll)
            lg, lt = mgt.epsg3031toepsg4326(lg, lt)

            dataices = {'date': data_img,
                        'latitude': lt,
                        'longitude': lg,
                        'minoraxis_px': int(menor_eixo[i-1]),
                        'majoraxis_px': int(maior_eixo[i-1]),
                        'area_px': int(areas[i-1]),
                        'perimeter_px': int(perimetro[i-1]),
                        'shape': morfo_track_1[i-1]}
        
            ices_df = ices_df.append(dataices, ignore_index=True)
    
    return ices_df