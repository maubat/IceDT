import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import perceptron
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

def getTreinoFeats_picks(path_IDB):

    train_feats = []
    for image_feats in sorted(os.listdir(path_IDB)):
        if image_feats.endswith(".txt"):
            fo = open(path_IDB + image_feats, "r")
            line = fo.readline()
            line = np.fromstring(line, dtype=float, sep=' ')
            train_feats.append(line)
            fo.close()

    train_feats = np.asarray(train_feats)

    return train_feats

def getTreinoFeats_DB(path_IDB):

    fo = open(path_IDB, "r")
    db = []
    for line in fo.readlines():
        line = np.fromstring(line, dtype=float, sep=' ')
        db.append(line)
    fo.close()

    treino = np.asarray(db)

    return treino

def saveTreinoSet(path_IDB, path_saida):

    file = open(path_saida + 'TrainTemp.txt', 'a')
    # for image_feats in sorted(os.listdir(path_IDB)):
    for image_feats in os.listdir(path_IDB):
        if image_feats.endswith(".txt"):
            fo = open(path_IDB + image_feats, "r")
            # ll = fo.readline()
            # print ll[-2:]
            # sys.exit(0)
            file.write(fo.readline()+'\n')
    file.close()

    file = open(path_saida + 'TrainSet.txt', 'a')
    with open(path_saida + 'TrainTemp.txt') as f:
        for line in f:
            if not line.isspace():
                file.write(line)
    file.close()
    os.remove(path_saida + 'TrainTemp.txt')

def saveTreinoSet_Seasonal(path_IDB, path_saida):

    file1 = open(path_saida + 'TVerao.txt', 'a')
    file2 = open(path_saida + 'TOutono.txt', 'a')
    file3 = open(path_saida + 'TInverno.txt', 'a')
    file4 = open(path_saida + 'TPrimavera.txt', 'a')

    for image_feats in sorted(os.listdir(path_IDB)):


        if image_feats.endswith(".txt"):

            data = image_feats[18:26]
            ano = data[0:4]
            mes = data[4:6]
            dia = data[6:]


            fo = open(path_IDB + image_feats, "r")

            if int(mes) >= 1 and int(mes) <= 3:
                # print 'verao'
                file1.write(fo.readline() + '\n')

            if int(mes) >= 4 and int(mes) <= 6:
                # print 'Outono'
                file2.write(fo.readline() + '\n')

            if int(mes) >= 7 and int(mes) <= 9:
                # print 'Inverno'
                file3.write(fo.readline() + '\n')

            if int(mes) >= 10 and int(mes) <= 12:
                # print 'Primavera'
                file4.write(fo.readline() + '\n')

    file1.close()
    file2.close()
    file3.close()
    file4.close()


    for i in range(1,5):
        print (i)
        if i == 1:
            estacao = 'TVerao.txt'
            file = open(path_saida + 'Verao.txt', 'a')
        if i == 2:
            estacao = 'TOutono.txt'
            file = open(path_saida + 'Outono.txt', 'a')
        if i == 3:
            estacao = 'TInverno.txt'
            file = open(path_saida + 'Inverno.txt', 'a')
        if i == 4:
            estacao = 'TPrimavera.txt'
            file = open(path_saida + 'Primavera.txt', 'a')

        with open(path_saida + estacao) as f:
            for line in f:
                if not line.isspace():
                    file.write(line)
        file.close()
        os.remove(path_saida + estacao)

def settreino(treino):

    # Treinamento Supervisionado
    # X Contem os descritores das Amostras
    x = treino[:, :-1]
    # Y Contem os labels referentes as amostras
    y = treino[:, -1].astype(int)


    rforest = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rforest.fit(x, y)

    csvm = svm.SVC(kernel='rbf')
    csvm.fit(x,y)

    percep = perceptron.Perceptron(n_iter=300, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
    percep.fit(x, y)

    knn = neighbors.KNeighborsClassifier(algorithm="auto")
    knn.fit(x,y)

    adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5,algorithm="SAMME")
    adab.fit(x,y)

    return rforest, csvm, percep, knn, adab

def settreino_ForestComite(treino, treino_fs):

    # Treinamento Supervisionado
    # X Contem os descritores das Amostras
    x = treino[:, :-1]
    # Y Contem os labels referentes as amostras
    y = treino[:, -1].astype(int)

    #Spectral
    rf1 = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf1.fit(x[:,0:4], y)

    #HistogramGabor
    rf2 = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf2.fit(x[:,4:13], y)

    #GLCM
    rf4 = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf4.fit(x[:,13:29], y)
    
    #Morfo
    rf5 = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf5.fit(x[:,29:38], y)

    #FS
    xfs = treino_fs[:, :-1]
    yfs = treino_fs[:, -1].astype(int)

    r_fs = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    r_fs.fit(xfs, yfs)

    return rf1,rf2,rf4,rf5,r_fs

def settreino_RF(treino):

    # Treinamento Supervisionado
    # X Contem os descritores das Amostras
    x = treino[:, :-1]
    # Y Contem os labels referentes as amostras
    y = treino[:, -1].astype(int)


    #rforest = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rforest = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                     max_depth=None, min_samples_split=2, random_state=0,
                                     class_weight='balanced')
    rforest.fit(x, y)

    #scores = cross_val_score(rforest, x, y, cv=30)

    return rforest
