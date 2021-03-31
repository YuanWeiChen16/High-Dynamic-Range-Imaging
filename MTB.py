import numpy as np

#turn img to gray
def gray(img):
    return np.array([[(54*yi[2]+183*yi[1]+19*yi[0])/256 for yi in xi] for xi in img], dtype='uint8')

#turn in to half size img
def shrinkBy2(img):
    retImg = np.array([[(int(img[i][j])+int(img[i][j+1])+int(img[i+1][j])+int(img[i+1][j+1]))/4            for j in range(0, img.shape[1]-1, 2)]            for i in range(0, img.shape[0]-1, 2)] , dtype='uint8')
    return retImg

#create threshold map
def bitmap(img):
    med = int(np.median(img))
    thresBitmap = np.array([[True if yi > med else False for yi in xi] for xi in img], dtype='bool')
    x, y = img.shape
    excluBitmap = np.full((x, y), True, dtype='bool')
    for i in range(x):
        for j in range(y):
            if abs(img[i][j] - med) < 5:
                excluBitmap[i][j] = False
    return (thresBitmap, excluBitmap)

#shift gray img 
def bitmapShift(bm, x, y):
    shifted = np.full(bm.shape, False, dtype='bool')
    if x > 0:
        shifted[x:] = bm[:-x]
    elif x < 0:
        shifted[:x] = bm[-x:]
    else:
        shifted = bm
    if y > 0:
        shifted = [np.append([False] * y, row[:-y]) for row in shifted]
    elif y < 0:
        shifted = [np.append(row[-y:], [False] * -y) for row in shifted]
    return shifted

#shift RGB img
def imgShift(im, x, y):
    shifted = np.full(im.shape, 0, dtype='uint8')
    if x > 0:
        shifted[x:] = im[:-x]
    elif x < 0:
        shifted[:x] = im[-x:]
    else:
        shifted = im
    if y > 0:
        shifted = [np.concatenate([[[0, 0, 0]] * y, row[:-y]]) for row in shifted]
    elif y < 0:
        shifted = [np.concatenate([row[-y:], [[0, 0, 0]] * -y]) for row in shifted]
    return shifted

#find Great Shift
def getExpShift(img0, img1):
    smlImg0 = shrinkBy2(img0)
    smlImg1 = shrinkBy2(img1)
    curShiftBits = getExpShift(smlImg0, smlImg1, 3)
    curShiftBits[0] *= 2
    curShiftBits[1] *= 2

    tb0, eb0 = bitmap(img0)
    tb1, eb1 = bitmap(img1)
    minErr = img0.shape[0] * img0.shape[1]

    for i in range(-1, 2):
        for j in range(-1, 2):
            xs = curShiftBits[0] + i
            ys = curShiftBits[1] + j
            shifted_tb1 = bitmapShift(tb1, xs, ys)
            shifted_eb1 = bitmapShift(eb1, xs, ys)
            diff_b = np.logical_xor(tb0, shifted_tb1)
            diff_b = np.logical_and(diff_b, eb0)
            diff_b = np.logical_and(diff_b, shifted_eb1)
            err = np.sum(diff_b)
            if err < minErr:
                ret = [xs, ys]
                minErr = err
    return ret

#Align img 
def align(img0, img1):
    g0 = gray(img0)
    g1 = gray(img1)
    return getExpShift(g0, g1)


def MTB(imgs_src):
    for i in range(1, len(imgs_src)):
        x, y = align(imgs_src[0], imgs_src[1])
        ret.append(imgShift(imgs_src[i], x, y))
    return ret


#use this like   images = MTB.MTB(images)