# -*- coding: utf-8 -*-

import cv2
import glob
import numpy as np
from __init__ import detector, predictor, Square

teeth_lower = np.array([0, 0, 80])
teeth_upper = np.array([180, 120, 255])

leftmostidx  = np.array([0, 12])
rightmostidx = np.array([6, 16])
uppermostidx = np.array([0, 6, 12, 13, 14, 15, 16])
lowermostidx = np.array([0, 6, 12, 16, 17, 18, 19])

upperlipidx  = np.array([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16])
lowerlipidx  = np.array([0, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19])

def select_proxy():  
    # select and generate teeth proxy frame(s)
    frs = 690, 2343     # manually selected
    tags = ('upper', 'lower')
    tar_path = 'target/target001.mp4'
    cap = cv2.VideoCapture(tar_path)
    
    for fr, tag in zip(frs, tags):
        # get face and landmarks
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, img = cap.read()
        cv2.imwrite('reference/proxy_%s_%04d.png' % (tag, fr), img)
        
def process_proxy(mode, rsize=300, ksize=(7, 7), sigma=1, k=2):
    # process teeth proxies to get their landmarks and high-pass filters
    assert(mode == 'upper' or mode == 'lower')
    pxyfile = glob.glob('reference/proxy_%s_*.png' % mode)[0]
    img = cv2.imread(pxyfile)
    
    # detect face and landmarks
    det = detector(img, 1)[0]
    shape = predictor(img, det)
    ldmk = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(48, shape.num_parts)], np.float32)
    
    # normalize landmarks
    origin = np.array([det.left(), det.top()])
    size = np.array([det.width(), det.height()])
    ldmk = (ldmk - origin) / size         # restrained in [0, 0] ~ [1, 1]
    
    # resize texture
    txtr = img[origin[1]:origin[1]+size[1], origin[0]:origin[0]+size[0]]
    txtr = cv2.resize(txtr, (rsize, rsize))
    
    # generate hgih-pass filter
    norm_txtr   = txtr.astype(np.float) / 255
    smooth_txtr = cv2.GaussianBlur(norm_txtr, ksize, sigma)
    filt   = (norm_txtr - smooth_txtr) * k + 0.5
    filt[filt < 0] = 0
    filt[filt > 1] = 1 
    
    return ldmk, filt

def detect_region(inpI, inpS, rsize, boundary):
    # automatically detect upper and lower teeth region in input image
    # boundary: (left, right, upper, lower)
    hsv = cv2.cvtColor(inpI, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, teeth_lower, teeth_upper)
    
    # eliminate region that is out of mouth
    alpha = 0.05
    left  = np.min(inpS[leftmostidx,  0]) + alpha
    right = np.max(inpS[rightmostidx, 0]) - alpha
    upper = np.min(inpS[uppermostidx, 1])
    lower = np.max(inpS[lowermostidx, 1])
    
    left  = int(round(left  * rsize - boundary[0]))
    right = int(round(right * rsize - boundary[0]))
    upper = int(round(upper * rsize - boundary[2]))
    lower = int(round(lower * rsize - boundary[2]))

    mask[:, :left]  = 0
    mask[:, right:] = 0   
    mask[:upper, :] = 0
    mask[lower:, :] = 0

#    cv2.imshow('raw',  inpI)    
#    for i in range(inpI.shape[2]):
#        inpI[:, :, i] &= mask
#    cv2.imshow('mask', inpI)
#    cv2.waitKey(0)
    
    # get teeth region
    ys, xs = mask.nonzero()
    region = np.array([[y, x] for y, x in zip(ys, xs)])
    
    if region.shape[0] == 0:
        # no region detected; return empty arrays
        upper_region, lower_region = np.array([]), np.array([])
    else:
        # divide region into upper and lower ones
        axis = (np.max(region[:, 0]) + np.min(region[:, 0])) / 2
        check = region[:, 0] <= axis
        upper_region = region[check.nonzero()]
        lower_region = region[np.logical_not(check).nonzero()]
        
    return upper_region, lower_region
    
def teeth_enhancement(inpI, inpS, pxyF, pxyS, region, rsize, boundary, mode):
    # enhance quality of input image given teeth proxy filter and their landmarks
    # region is the part which is to be enhanced
    assert(mode == 'upper' or mode == 'lower')
    
    # calculate displacement between input image and proxy
    inpS_half = inpS[upperlipidx, :] if mode == 'upper' else inpS[lowerlipidx, :]
    pxyS_half = pxyS[upperlipidx, :] if mode == 'upper' else pxyS[lowerlipidx, :]
    dx, dy = np.round(np.mean(inpS_half - pxyS_half, axis=0) * rsize).astype(np.int)
    
    outpI = np.copy(inpI)
    x_bd, y_bd = boundary[0], boundary[2]
    for (y_pt, x_pt) in region:
        for c in range(inpI.shape[2]):
            # in each channel
            # (x_pxy, y_pxy) in proxy -> (x_pt, y_pt) in input image
            x_pxy = x_bd + x_pt - dx
            y_pxy = y_bd + y_pt - dy
            if pxyF[y_pxy, x_pxy, c] < 0.5:
                outpI[y_pt, x_pt, c] *= 2 * pxyF[y_pxy, x_pxy, c]
            else:
                outpI[y_pt, x_pt, c] = 255 - 2*(255-outpI[y_pt, x_pt, c])*(1-pxyF[y_pxy, x_pxy, c])
    
    return outpI

def sharpen(inpI, ksize=(7, 7), sigma=1, k=0.5):
    smooth_inpI = cv2.GaussianBlur(inpI, ksize, sigma)
    outpI = (inpI - smooth_inpI)*k + inpI
    return outpI

def test1():
    from candidate import weighted_median
    # load target data  
    tgtdata = np.load('target/target001.npz')
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    
    # clip target textures
    sq = Square(0.25, 0.75, 0.6, 1.00)
    boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
    
    # load input data
    inpdata = np.load('input/test036_ldmks.npy')[20:]
    
    # load proxy landmarks and filters
    pxySU, pxyFU = process_proxy('upper')
    pxySL, pxyFL = process_proxy('lower')
    
    # create every frame and form a mp4
    for cnt, inpS in enumerate(inpdata):
        tmpI, tmpS = weighted_median(inpS, tgtS, tgtI, 50)        
        regionU, regionL = detect_region(tmpI, tmpS, 300, boundary)
        
        outpI, outpS = np.copy(tmpI), np.copy(tmpS)
        for mode in ('upper', 'lower'):
            pxyS = pxySU if mode == 'upper' else pxySL
            pxyF = pxyFU if mode == 'upper' else pxyFL
            region = regionU if mode == 'upper' else regionL
            outpI = teeth_enhancement(outpI, outpS, pxyF, pxyS, region, 300, boundary, mode)

        cv2.imshow('raw', tmpI)        
        cv2.imshow('enhance', outpI)
        cv2.waitKey(0)
        diffI = outpI - tmpI
        print(np.sum(diffI != 0))
    
if __name__ == '__main__':
    test1()