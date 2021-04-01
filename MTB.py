import numpy as np

#turn img to gray
def gray(img):
    return np.array([[(54*yi[2]+183*yi[1]+19*yi[0])/256 for yi in xi] for xi in img], dtype=int)

#turn in to half size img
def Smallimg(img):
    smalimg = np.array([[(int(img[i][j])+int(img[i][j+1])+int(img[i+1][j])+int(img[i+1][j+1]))/4            for j in range(0, img.shape[1]-1, 2)]            for i in range(0, img.shape[0]-1, 2)] , dtype='uint8')
    return smalimg

#create threshold map
def bitmap(img):
    #find midThreshold
    med = int(np.median(img))
    thresBitmap = np.array([[True if yi > med else False for yi in xi] for xi in img], dtype = bool)

    x, y = img.shape
    excluBitmap = np.full((x, y), True, dtype=bool)
    #cut mid error
    for i in range(x):
        for j in range(y):
            if abs(img[i][j] - med) < 4:
                excluBitmap[i][j] = False
    return (thresBitmap, excluBitmap)

#shift Bit img 
def bitmapShift(bm, x, y):
    shifted = np.full(bm.shape, False, dtype=bool)
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
    shifted = np.full(im.shape, 0, dtype=int)
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
def getExpShift(img0, img1, shiftBits):
    if shiftBits > 0:
        smlImg0 = Smallimg(img0)
        smlImg1 = Smallimg(img1)
        curShiftBits = getExpShift(smlImg0, smlImg1, shiftBits - 1)
        curShiftBits[0] *= 2
        curShiftBits[1] *= 2
    else:
        curShiftBits = [0, 0]
    #create bit map
    tb0, eb0 = bitmap(img0)
    tb1, eb1 = bitmap(img1)
    #culc error
    minErr = img0.shape[0] * img0.shape[1]

    #-1 ~ 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            xs = curShiftBits[0] + i
            ys = curShiftBits[1] + j
            shifted_tb1 = bitmapShift(tb1, xs, ys)
            shifted_eb1 = bitmapShift(eb1, xs, ys)
            
            #MTB logic   (img XOR shiftimg) AND mask 
            diff_b = np.logical_xor(tb0, shifted_tb1)
            diff_b = np.logical_and(diff_b, eb0)
            diff_b = np.logical_and(diff_b, shifted_eb1)
            err = np.sum(diff_b)
            if err < minErr:
                ret = [xs, ys]
                minErr = err
    return ret

#Align img 
def align(img0, img1, level):
    g0 = gray(img0)
    g1 = gray(img1)
    return getExpShift(g0, g1, level)


def MTB(imgs_src, level):
    ret = [imgs_src[0]]
    for i in range(1, len(imgs_src)):
        #find Greate Shift
        x, y = align(imgs_src[0], imgs_src[1], level)
        #shift
        ret.append(imgShift(imgs_src[i], x, y))
    return ret


#use this like   images = MTB.MTB(images)