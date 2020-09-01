

def image_percentil(f, perc):

    import numpy as np

    v = np.zeros((1, len(perc)))
    i=0
    for p in perc:

        k = (f.size-1) * p/100.
        dw = np.floor(k).astype(int)
        up = np.ceil(k).astype(int)
        g  = np.sort(f.ravel())
        d  = g[dw]
        d0 =   d   * (up-k)
        d1 = g[up] * (k -dw)
        v[0][i] = np.where(dw==up, d, d0+d1)
        i+=1
    return v

def hist_percentil(h,p):

    import numpy as np
    s = h.sum()
    k = ((s-1) * p/100.)+1
    dw = np.floor(k)
    up = np.ceil(k)
    hc = np.cumsum(h)
    if isinstance(p, int):
       k1 = np.argmax(hc>=dw)
       k2 = np.argmax(hc>=up)
    else:
       k1 = np.argmax(hc>=dw[:,np.newaxis],axis=1)
       k2 = np.argmax(hc>=up[:,np.newaxis],axis=1)
    d0 = k1 * (up-k)
    d1 = k2 * (k -dw)
    return np.where(dw==up,k1,d0+d1)