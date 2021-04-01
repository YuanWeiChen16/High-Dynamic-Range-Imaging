import matplotlib.pyplot as plt
import numpy as np
import random
import math
import cv2
import os

def weight(z):
    z_max, z_min = 255., 0. 
    if z <= ((z_max + z_min) / 2):
        return z - z_min
    else:
        return z_max - z

def estimate_curve(Z, exps, L):
    z_max, z_min = 255., 0. 
    n = 255
    A = np.zeros(shape = ((Z.shape[0] * Z.shape[1] + n, n + Z.shape[0] + 1)), dtype=float)
    b = np.zeros(shape = (A.shape[0], 1), dtype = float)

    k = 0
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            wij = weight(Z[i][j])
            A[k][Z[i][j]] = wij
            A[k][n + i + 1] = -wij
            b[k][0] = wij * exps[j]
            k = k + 1

    A[k][128] = 1
    k = k + 1

    for i in range(0, n - 1):
        A[k][i] = L * weight(i + 1)
        A[k][i + 1] = -2 * L * weight(i + 1)
        A[k][i + 2] = L * weight(i + 1)
        k = k + 1
    
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    g = x[:256]
    lE = x[256:]
    #print(g)
    return g[:,0]

def estimate_radiance(imgs, exps, curve):
    img_shape = imgs.shape
    rad = np.zeros(shape=img_shape[1:], dtype=float)

    imgNums = imgs.shape[0]

    for i in range(0, img_shape[1]):
        for j in range(0, img_shape[2]):
            g = np.ndarray(shape = imgNums, dtype=float)
            w = np.ndarray(shape = imgNums, dtype=float)
            for k in range(0, imgNums):
                g[k] = curve[int(imgs[k][i][j])]
                w = weight(imgs[k][i][j])

            sumOfW = np.sum(w)
            if sumOfW > 0:
                rad[i][j] = np.sum(w * (g - exps) / sumOfW)
            else:
                rad[i][j] = np.sum(w * (g - exps))
    return rad

def hdr_debvec(Z, img, exps, L, show):
    print("---------Using Debvec Method to estimate radiance---------")
    hdr_img = np.ndarray(shape = (img[0].shape), dtype = float)
    rad = np.ndarray(shape = (img[0].shape), dtype = float)
    plt.figure(figsize=(10, 10))
    for i in range(0, 3):
        g = estimate_curve(Z[i], exps, L)
        plt.plot(g, range(256), i)
        rad[:,:,i] = estimate_radiance(img[:,:,:,i], exps, g)
        hdr_img[:,:,i] = cv2.normalize(rad[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    if show:
        plt.show()
    print("---------Finished estimating radiance---------")
    return rad, hdr_img

def Tone(img_src):
    #max l
    LMax = (int)(np.max(img_src))
    #create tone map img
    ToneImg = np.full(img_src.shape, 0, dtype=int)
    
    #best easy way to do this
    x, y = img_src.shape
    for i in range(x):
        for j in range(y):
           ToneImg[i][j] = (int)((img_src[i][j] /(LMax + 1))*255)
    return ToneImg

def load(path_test):
    filenames = []
    exposure_times = []
    f = open(os.path.join(path_test, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        # (filename, exposure, *rest) = line.split()
        (filename, exposure) = line.split()
        filenames += [os.path.join(path_test,filename)]
        # exposure_times += [math.log(float(exposure),2)]
        exposure_times += [float(exposure)]
    return filenames, exposure_times

def read(files):
    imageNum = len(files)
    img_shape = cv2.imread(files[0]).shape
    x_max = img_shape[0]
    y_max = img_shape[1]

    img = np.ndarray(shape = (imageNum, x_max, y_max, 3), dtype = int)
    for i in range(len(files)):
        img[i] = cv2.imread(files[i])
    
    return x_max, y_max, img

def sampleImage(x_max, y_max, img, sampleNum, imageNum, mode):
    random.seed(a = None, version = 2)
    idx = np.random.randint(x_max, size = sampleNum)
    idy = np.random.randint(y_max, size = sampleNum)
    #index = np.random.uniform(low = [0, 0], high = [x_max, y_max], size = (sampleNum, 2))

    Z = np.ndarray(shape = (3, sampleNum, imageNum), dtype = int)

    
    if mode == 'random':
        print("---------Using random to sample pixel---------")
        for i in range(0, sampleNum):
            for j in range(0, imageNum):
                Z[0][i][j] = img[j][idx[i]][idy[i]][0]
                Z[1][i][j] = img[j][idx[i]][idy[i]][1]
                Z[2][i][j] = img[j][idx[i]][idy[i]][2]
    elif mode == 'uniform':
        print("---------Using uniform to sample pixel---------")
        pos = []
        h_step, w_step = x_max // (sampleNum + 1), y_max // (sampleNum + 1)
        for i in range(1, sampleNum + 1):
            for j in range(1, sampleNum + 1):
                pos.append((i * h_step, j * w_step))
        for i, (x, y) in enumerate(pos):
            for j in range(imageNum):
                if i < 256:
                    Z[0][i][j] = img[j][x][y][0]
                    Z[1][i][j] = img[j][x][y][1]
                    Z[2][i][j] = img[j][x][y][2]

    return Z

def computeRadianceMap(img_src, path):
    
    from matplotlib.pylab import cm
    colorize = cm.jet
    cmap = np.float32(cv2.cvtColor(np.uint8(img_src), cv2.COLOR_BGR2GRAY)/255.)
    cmap = colorize(cmap)
    cv2.imwrite(os.path.join(path, 'cmap.jpg'), np.uint8(cmap*255.))

def computeHDRImg(rad, hdr_nm, path):

    cv2.imwrite(os.path.join(path, 'hdr_nm.jpg'), hdr_nm)
    hdr_tm = np.ndarray(shape = (hdr_nm.shape), dtype = float)
    for i in range(0,3):
        hdr_tm[:,:,i] = Tone(rad[:,:,i])

    cv2.imwrite(os.path.join(path, 'hdr_tm.jpg'), hdr_tm)
