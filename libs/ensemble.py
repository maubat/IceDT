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

def doEnsemble(committee, feats_to_class, conf=0.65, trust_lvl=0.92, harmony_lvl=0.5):

    c1 = committee[0]
    c2 = committee[1]
    c4 = committee[2]
    c5 = committee[3]
    cfull = committee[4]
    cfs = committee[5]

    votes = []
    v_mp = 0
    pesos = 0

    count_v = 0
    count_vp = 0

    scfull = cfull.predict_proba(feats_to_class)
    if scfull[0, 0] < conf:
        v = 2
        v_mp += v * scfull[0, 1]
        pesos += scfull[0, 1]
    else:
        v = 1
        v_mp += v * scfull[0, 0]
        pesos += scfull[0, 0]

    votes.append(v)

    sc1 = c1.predict_proba(feats_to_class[:, 0:4])
    if sc1[0, 0] < conf:
        v = 2
        v_mp += v * sc1[0, 1]
        pesos += sc1[0, 1]
        if sc1[0, 1] > trust_lvl:
            v_mp += v * sc1[0, 1]
            pesos += sc1[0, 1]
    else:
        v = 1
        v_mp += v * sc1[0, 0]
        pesos += sc1[0, 0]

    votes.append(v)

    sc2 = c2.predict_proba(feats_to_class[:, 4:13])
    if sc2[0, 0] < conf:
        v = 2
        v_mp += v * sc2[0, 1]
        pesos += sc2[0, 1]
    else:
        v = 1
        v_mp += v * sc2[0, 0]
        pesos += sc2[0, 0]

    votes.append(v)

    if sc2[0, 1] > trust_lvl:
        votes.append(v)

    sc4 = c4.predict_proba(feats_to_class[:, 13:29])
    if sc4[0, 0] < conf:
        v = 2
        v_mp += v * sc4[0, 1]
        pesos += sc4[0, 1]
    else:
        v = 1
        v_mp += v * sc4[0, 0]
        pesos += sc4[0, 0]

    votes.append(v)

    if sc4[0, 1] > trust_lvl:
        votes.append(v)
        votes.append(v)

    sc5 = c5.predict_proba(feats_to_class[:, 29:38])
    if sc5[0, 0] < 0.51:
        v = 2
        v_mp += v * sc5[0, 1]
        pesos += sc5[0, 1]
        if sc5[0, 1] > 0.9:
            v_mp += v * sc5[0, 1]
            pesos += sc5[0, 1]
    else:
        v = 1
        v_mp += v * sc5[0, 0]
        pesos += sc5[0, 0]
        v_mp += v * sc5[0, 0]
        pesos += sc5[0, 0]
        if sc5[0, 0] > 0.9:
            v_mp += v * sc5[0, 0]
            pesos += sc5[0, 0]

    votes.append(v)
    votes.append(v)

    if sc5[0, 0] > 0.9 or sc5[0, 1] > 0.9:
        votes.append(v)
        votes.append(v)

    feats_fs = np.zeros((1, 12))
    feats_fs[0, 0] = feats_to_class[:, 0]
    feats_fs[0, 1] = feats_to_class[:, 4]
    feats_fs[0, 2] = feats_to_class[:, 6]
    feats_fs[0, 3] = feats_to_class[:, 7]
    feats_fs[0, 4] = feats_to_class[:, 11]
    feats_fs[0, 5] = feats_to_class[:, 12]
    feats_fs[0, 6] = feats_to_class[:, 14]
    feats_fs[0, 7] = feats_to_class[:, 16]
    feats_fs[0, 8] = feats_to_class[:, 25]
    feats_fs[0, 9] = feats_to_class[:, 30]
    feats_fs[0, 10] = feats_to_class[:, 31]
    feats_fs[0, 11] = feats_to_class[:, 36]

    sc6 = cfs.predict_proba(feats_fs)
    if sc6[0, 0] < conf:  # conf
        v = 2
        v_mp += v * sc6[0, 1]
        pesos += sc6[0, 1]

    else:
        v = 1
        v_mp += v * sc6[0, 0]
        pesos += sc6[0, 0]

    votes.append(v)

    if sc6[0, 1] > trust_lvl:
        votes.append(v)

    v = majorityVote(votes)
    vp = int(round(v_mp / pesos))

    if v == 1: count_v += 1
    if vp == 1: count_vp += 1

    if v == 1 and vp == 1:
        vf = 1
    else:
        vf = 2

    score_prediction = (sc1 + sc2 + sc4 + sc5 + sc6 + scfull) / 6.0

    #vf = vp
    if vf == 1 and score_prediction[0, 0] < harmony_lvl: #Use when the committee is too divergent
        vf = 2

    vf = vp
    result = [vf, score_prediction, count_v, count_vp]

    # if vf == 1:
    #     print('Iceberg detected.')
    #     print('Votes: ', votes, 'Harmony: ', round(score_prediction[0, 0], 2))
    #     print('confidence: ')
    #     print(round(sc1[0, 0], 2), round(sc2[0, 0], 2), round(sc4[0, 0], 2),
    #           round(sc5[0, 0], 2), round(sc6[0, 0], 2), round(scfull[0, 0], 2))
    # else:
    #     print('NON iceberg.')
    #     print('Votes: ', votes, 'Harmony: ', round(score_prediction[0, 1], 2))
    #     print('confidence: ')
    #     print(round(sc1[0, 1], 2), round(sc2[0, 1], 2), round(sc4[0, 1], 2),
    #           round(sc5[0, 1], 2), round(sc6[0, 1], 2), round(scfull[0, 1], 2))

    return result
