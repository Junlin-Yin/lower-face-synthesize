# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import cv2
import glob
import numpy as np
from __init__ import detector, predictor, Square

teeth_hsv_lower  = np.array([0, 0, 60])
teeth_hsv_upper  = np.array([180, 150, 255])

uppermostidx = np.array([12, 13, 14, 15, 16])
lowermostidx = np.array([12, 19, 18, 17, 16])

upperlipidx  = np.array([1, 2, 3, 4, 5, 13, 14, 15])
lowerlipidx  = np.array([7, 8, 9, 10, 11, 17, 18, 19])

def select_proxy(tar_path, mode, fr):  
    # select and generate teeth proxy frame(s)
    assert(mode == 'upper' or mode == 'lower')
    cap = cv2.VideoCapture(tar_path)
    # get face and landmarks
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ret, img = cap.read()
    pxyfile = glob.glob('reference/proxy_%s_*.png' % mode)
    for f in pxyfile:
        os.remove(f)
    cv2.imwrite('reference/proxy_%s_%04d.png' % (mode, fr), img)

def process_proxy(thresholdU, thresholdL, rsize, ksize=(17,17), sigma=1e2, k=1):
    # process teeth proxies to get their landmarks and high-pass filters
    F, S = {}, {}
    for mode in ('upper', 'lower'):
        threshold = thresholdU if mode == 'upper' else thresholdL
        
        pxyfile, = glob.glob('reference/proxy_%s_*.png' % mode)
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
        
        # generate hgih-pass filter (only one channel)
        norm_txtr   = txtr.astype(np.float) / 255
        smooth_txtr = cv2.GaussianBlur(txtr, ksize, sigma) / 255
        filt   = (norm_txtr - smooth_txtr) * k + 0.5
        filt[filt < 0] = 0
        filt[filt > 1] = 1
        
        # 'clip' the other side
        if mode == 'upper':
            filt[threshold:, :] = np.min(filt[threshold, :])
        else:
            filt[:threshold, :] = np.min(filt[threshold, :])
        
        # add landmarks and filter into dict S and F respectively
        F[mode] = filt
        S[mode] = ldmk
        
    return F, S

def detect_region(inpI, inpS, rsize, boundary):
    # automatically detect upper and lower teeth region in input image
    # boundary: (left, right, upper, lower)
    upper_xp = inpS[uppermostidx][:, 0] * rsize - boundary[0]
    upper_fp = inpS[uppermostidx][:, 1] * rsize - boundary[2]
    lower_xp = inpS[lowermostidx][:, 0] * rsize - boundary[0]
    lower_fp = inpS[lowermostidx][:, 1] * rsize - boundary[2]
    upper_bd = np.ceil(np.interp(np.arange(rsize), upper_xp, upper_fp)).astype(np.int)
    lower_bd = np.ceil(np.interp(np.arange(rsize), lower_xp, lower_fp)).astype(np.int)
    
    hsv = cv2.cvtColor(inpI, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, teeth_hsv_lower, teeth_hsv_upper)
    xs, ys = mask.nonzero()
    region = np.array([(y, x) for (x, y) in zip(xs, ys)])
    if region.shape[0] == 0:
        # no region detected; return empty arrays
        upper_region, lower_region = np.array([]), np.array([])
        return upper_region, lower_region
        
    check  = np.logical_and(region[:, 0] > upper_bd[region[:, 1]],
                            region[:, 0] < lower_bd[region[:, 1]])
    region = region[check.nonzero()]
        
    if region.shape[0] == 0:
        # no region detected; return empty arrays
        upper_region, lower_region = np.array([]), np.array([])
    else:
        # divide region into upper and lower ones
        lmda = 0.6
        axis = lmda*np.max(region[:, 0]) + (1-lmda)*np.min(region[:, 0])
        check = region[:, 0] <= axis
        upper_region = region[check.nonzero()]
        lower_region = region[np.logical_not(check).nonzero()]
        
    return upper_region, lower_region
    
def local_enhancement(inpI, inpS, pxyF, pxyS, region, rsize, boundary, mode):
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
        # (x_pxy, y_pxy) in proxy -> (x_pt, y_pt) in input image
        x_pxy = x_bd + x_pt - dx
        y_pxy = y_bd + y_pt - dy
        for c in range(outpI.shape[2]):
            if pxyF[y_pxy, x_pxy, c] < 0.5:
                outpI[y_pt, x_pt, c] = 2*pxyF[y_pxy, x_pxy, c] * outpI[y_pt, x_pt, c]
            else:
                outpI[y_pt, x_pt, c] = 255 - 2*(255-outpI[y_pt, x_pt, c])*(1-pxyF[y_pxy, x_pxy, c])
    return outpI

def sharpen(inpI, ksize=(15, 15), sigma=2e1, k=0.5):
    smooth_inpI = cv2.GaussianBlur(inpI, ksize, sigma)
    outpI = (inpI.astype(np.float) - smooth_inpI.astype(np.float))*k + inpI.astype(np.float)
    outpI[outpI < 0]   = 0
    outpI[outpI > 255] = 255
    outpI = outpI.astype(np.uint8)
    return outpI

def process_teeth(inpI, inpS, pxyF, pxyS, rsize, boundary):
    regionU, regionL = detect_region(inpI, inpS, rsize, boundary)
    # enhance upper region
    tmpI = local_enhancement(inpI, inpS, pxyF['upper'], pxyS['upper'], regionU, rsize, boundary, 'upper')
    # enhance lower region
    tmpI = local_enhancement(tmpI, inpS, pxyF['lower'], pxyS['lower'], regionL, rsize, boundary, 'lower')
    # sharpening
#    outpI = sharpen(tmpI)
    outpI = tmpI
    return outpI 

def test1():
    from candidate import weighted_median
    sq = Square(0.25, 0.75, 0.6, 1.00)
    n     = 100
    kzs   = [15, 17, 19, 21, 23, 25]
    sigmas= [1e1, 1e2, 2e2, 3e2, 4e2, 5e2]
    k     = 1
    rsize = 300
    tgtdata = np.load('target/target001.npz')
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
    inpS = np.load('input/test036_ldmks.npy')[25]
    _, H, W, C = tgtI.shape
    specI = np.zeros((H*len(sigmas), W*len(kzs), C))
    for y, sigma in enumerate(sigmas):
        for x, kz in enumerate(kzs):
            print('%g - %g' % (sigma, kz))
            pxyF, pxyS = process_proxy(thresholdU=213, thresholdL=235, rsize=rsize, ksize=(kz, kz), sigma=sigma, k=k)        
            tmpI, tmpS = weighted_median(inpS, tgtS, tgtI, n)
            outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, boundary)
            specI[y*H:y*H+H, x*W:x*W+W, :] = outpI
            if sigma == 1e2 and kz == 17:
                cv2.imwrite('reference/0025_teeth.png', outpI)
    cv2.imwrite('reference/spec_k=1.png', specI)
  
def test2():
    kz    = 15
    sigmas= [1, 2e1, 4e1, 6e1, 8e1, 1e2]
    ks    = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    inpI = cv2.imread('reference/0025_teeth.png')
    H, W, C = inpI.shape
    specI = np.zeros((H*len(sigmas), W*len(ks), C))
    for y, sigma in enumerate(sigmas):
        for x, k in enumerate(ks):  
            print('%g - %g' % (sigma, k))
            outpI = sharpen(inpI, ksize=(kz, kz), sigma=sigma, k=k)
            specI[y*H:y*H+H, x*W:x*W+W, :] = outpI
            if sigma == 2e1 and k == 0.5:
                cv2.imwrite('reference/0025_final.png', outpI)
    cv2.imwrite('reference/spec_sharp_kz%d.png'%kz, specI)
    
def test3():
    tar_path = 'target/target001.mp4'
    fr = 3261
    mode = 'upper'
    select_proxy(tar_path, mode, fr)

if __name__ == '__main__':
    test2()