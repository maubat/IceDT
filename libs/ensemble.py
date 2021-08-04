import numpy as np
import os

def majorityVote(votes):

    v_ice = 0
    n_ice = 0
    for j in votes:
        if j == 1:
            v_ice += 1
        else:
            n_ice += 1

    if v_ice > n_ice:
        v = 1
    else:
        v = 2

    return v

def majorityVote_class(votes,l):

    v_ice = 0
    n_ice = 0
    for j in range(0,l):
        if votes[j] == 1:
            v_ice += 1
        else:
            n_ice += 1

    if v_ice > n_ice:
        v = 1
    else:
        v = 0

    return v

def meanVotes(votes):

    v = int(round(np.mean(votes)))
    return v

def doEnsemble(committee, feats_to_class, conf=0.65, trust_lvl=0.92, harmony_lvl=0.5, deep_conf = 0.8):

    c5 = committee[0]

    votes = []
    v_mp = 0
    pesos = 0

    count_v = 0
    count_vp = 0

    sc5 = c5.predict_proba(feats_to_class[:, 29:38])
    if sc5[0, 0] < conf:
        voto_shape = 2
    else:
        voto_shape = 1

    score_prediction = sc5
    
    deep_mlp = committee[1]
    
    feats_fs = np.zeros((1, 9))
    feats_fs[0, 0] = feats_to_class[:, 0]
    feats_fs[0, 1] = feats_to_class[:, 1]
    feats_fs[0, 2] = feats_to_class[:, 3]
    feats_fs[0, 3] = feats_to_class[:, 8]
    feats_fs[0, 4] = feats_to_class[:, 9]
    feats_fs[0, 5] = feats_to_class[:, 20]
    feats_fs[0, 6] = feats_to_class[:, 31]
    feats_fs[0, 7] = feats_to_class[:, 33]
    feats_fs[0, 8] = feats_to_class[:, 34]
    
    v_deep = 2 if deep_mlp.predict(feats_fs)[0,0] < deep_conf else 1
    if v_deep == 1 and voto_shape == 1:
        vf = 1
    else:
        vf = 2

    result = [vf, score_prediction, count_v, count_vp]

    return result
