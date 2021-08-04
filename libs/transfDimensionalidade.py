import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

def transfDimens(train_feats, modo=0, dim=3):

    train_feats = np.asarray(train_feats)

    if modo == 0:

        print ('Redimensionado Features com PCA')
        # Analise de componentes principais (PCA)
        train_feats = np.asarray(train_feats)
        # print 'Feats Matrix ', train_feats.shape
        # print train_feats

        pca = PCA(n_components=dim)
        pca.fit(train_feats)
        new_feats = pca.transform(train_feats)

    else:
        new_feats = train_feats


    return new_feats