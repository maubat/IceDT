import os
import sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from datetime import datetime
from sklearn.model_selection import cross_val_score

from keras.models import Sequential, Model
from keras.layers import Dense, Input


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


def saveTreinoSet(path_IDB, path_saida):

    file = open(path_saida + 'TrainTemp.txt', 'a')
    for image_feats in os.listdir(path_IDB):
        if image_feats.endswith(".txt"):
            fo = open(path_IDB + image_feats, "r")
            file.write(fo.readline()+'\n')
    file.close()

    file = open(path_saida + 'TrainSet.txt', 'a')
    with open(path_saida + 'TrainTemp.txt') as f:
        for line in f:
            if not line.isspace():
                file.write(line)
    file.close()
    os.remove(path_saida + 'TrainTemp.txt')

    
def settreino_ForestComite(train):
    
    headnames = []
    for i in range(0, len(train.iloc[0, :-1])):
        headnames.append('feat'+str(i))

    headnames.append('target')
    train.columns = headnames
    
    x = train.iloc[:, :-1].values
    y = train.iloc[:, -1].values.astype(int)
    
    '''
    #Radiometrics
    rf1 = ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf1.fit(x[:,0:4], y)

    #HistogramGabor
    rf2 = ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf2.fit(x[:,4:13], y)

    #GLCM
    rf4 = ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf4.fit(x[:,13:29], y)
    '''
    
    #Morfo
    rf5 = ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0)
    rf5.fit(x[:,29:38], y)

    #return rf1,rf2,rf4,rf5
    return rf5

def set_deeplearning(train):
    
    headnames = []
    for i in range(0, len(train.iloc[0, :-2])):
        headnames.append('feat'+str(i))

    headnames.append('target')
    headnames.append('del')
    train.columns = headnames

    train.drop('del', axis='columns', inplace=True)

    train.loc[train.target == 2, 'target'] = 0
    
    train = train[['feat0', 'feat1', 'feat3', 'feat8', 'feat9', 'feat20', 'feat31', 'feat33', 'feat34', 'target']]
    
    X = train.iloc[:, :-1].values
    y = train.iloc[:, -1].values.astype(int)

    visible = Input(shape=(9,))
    hidden1 = Dense(10, activation='relu')(visible)
    hidden2 = Dense(20, activation='relu')(hidden1)
    hidden3 = Dense(10, activation='relu')(hidden2)
    output = Dense(1, activation='sigmoid')(hidden3)
    model_MLP = Model(inputs=visible, outputs=output)
    model_MLP.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model_MLP.fit(X, y, epochs=300, batch_size=1, verbose=0)
    
    #_, accuracy = model_MLP.evaluate(X, y)
    #print('Accuracy: %.2f' % (accuracy*100))
    
    return model_MLP
