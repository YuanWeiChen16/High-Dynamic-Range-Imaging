import HDR_utils as hdr
import MTB as mtb
import numpy as np
import argparse
sampleNum = 256

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default="")
parser.add_argument('--sample_mode', type=str, default="uniform")
parser.add_argument('--mtb_level', type=int, default=0)
parser.add_argument('--lambda_', type=int, default=50)
parser.add_argument('--showPlot', type=bool, default=False)
config = parser.parse_args()

files, exps = hdr.load(config.img_path)
x_max, y_max, img = hdr.read(files)
imageNum = len(files)

if config.mtb_level != 0:
    img = np.asarray(mtb.MTB(img, config.mtb_level))

Z = hdr.sampleImage(x_max, y_max, img, sampleNum, imageNum, config.sample_mode)

rad, hdr_img = hdr.hdr_debvec(Z, img, exps, config.lambda_, config.showPlot)

hdr.computeRadianceMap(hdr_img, config.img_path)
hdr.computeHDRImg(rad, hdr_img, config.img_path)
