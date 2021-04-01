import numpy as np

def Tone(img_src):
    #max l
    LMax = (int)np.max(img_src)
    #create tone map img
    ToneImg = np.full(img_src.shape, 0, dtype=int)
    
    #best easy way to do this
    x, y = img_src.shape
    for i in range(x):
        for j in range(y):
           ToneImg[i][j] = (int)((img_src[i][j] /(LMax + 1))*255)
    return ToneImg