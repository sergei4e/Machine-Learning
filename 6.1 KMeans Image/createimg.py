import numpy as np


def createx(img):
    x = []
    for i in img:
        for j in i:
            x.append(j)
    return np.array(x)


def create_img2(x, y, t):
    R = {x: [] for x in np.unique(y)}
    G = {x: [] for x in np.unique(y)}
    B = {x: [] for x in np.unique(y)}
    for n, i in enumerate(x):
        R[y[n]].append(i[0])
        G[y[n]].append(i[1])
        B[y[n]].append(i[2])

    if t == 'mean':
        for k in R:
            R[k] = np.mean(R[k])
            G[k] = np.mean(G[k])
            B[k] = np.mean(B[k])
    elif t == 'median':
        for k in R:
            R[k] = np.median(R[k])
            G[k] = np.median(G[k])
            B[k] = np.median(B[k])
    else:
        raise ValueError('type must be median or mean')

    x2 = np.array(x)
    for n, i in enumerate(x):
        x2[n] = np.array([R[y[n]], G[y[n]], B[y[n]]])
    img2 = []
    co = 0
    line = []
    for i in x2:
        line.append(i)
        co += 1
        if co == 713:
            img2.append(line)
            line = []
            co = 0
    return np.array(img2)
