# -*- coding: utf-8 -*-

import os
import cv2
import glob
import numpy as np
from __init__ import detector, predictor, Square

teeth_hsv_lower  = np.array([0, 0, 80])
teeth_hsv_upper  = np.array([180, 120, 255])

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

def process_proxy(thresholdU, thresholdL, biasU, biasL, rsize=300, ksize=(7, 7), sigma=1, k=0.5):
    # process teeth proxies to get their landmarks and high-pass filters
    F, S = {}, {}
    for mode in ('upper', 'lower'):
        threshold = thresholdU if mode == 'upper' else thresholdL
        bias = biasU if mode == 'upper' else biasL
        
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
        filt = (filt * 255).astype(np.uint8)
        filt = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
        filt = filt.astype(np.float) / 255
        
        # 'clip' the other side
        if mode == 'upper':
            filt[threshold:, :] = np.min(filt[threshold, :])
        else:
            filt[:threshold, :] = np.min(filt[threshold, :])
        # add a bias for adjustment
        filt += bias
        
        # add landmarks and filter into dict S and F respectively
        F[mode] = filt        
        S[mode] = ldmk
        
    return F, S

def detect_region(inpI, inpS, rsize, boundary, alpha, betaU, betaL):
    # automatically detect upper and lower teeth region in input image
    # boundary: (left, right, upper, lower)
    upper_xp = inpS[uppermostidx][:, 0] * rsize - boundary[0]
    upper_fp = inpS[uppermostidx][:, 1] * rsize - boundary[2]
    lower_xp = inpS[lowermostidx][:, 0] * rsize - boundary[0]
    lower_fp = inpS[lowermostidx][:, 1] * rsize - boundary[2]
    upper_bd = np.ceil(np.interp(np.arange(rsize), upper_xp, upper_fp)).astype(np.int)
    lower_bd = np.ceil(np.interp(np.arange(rsize), lower_xp, lower_fp)).astype(np.int)
    xs, = (lower_bd - upper_bd).nonzero()
    xs = xs[alpha:-alpha]
    region = np.array([(y, x) for x in xs for y in range(upper_bd[x]+betaU, lower_bd[x]-betaL)])
    
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
    
def local_enhancement(inpI, inpS, pxyF, pxyS, region, rsize, boundary, mode, dis):
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
        y_pxy = y_bd + y_pt - dy + dis
        if pxyF[y_pxy, x_pxy] < 0.5:
            outpI[y_pt, x_pt] = np.mean(2*pxyF[y_pxy, x_pxy] * outpI[y_pt, x_pt])
        else:
            outpI[y_pt, x_pt] = np.mean(255 - 2*(255-outpI[y_pt, x_pt])*(1-pxyF[y_pxy, x_pxy]))
    return outpI

def sharpen(inpI, ksize=(7, 7), sigma=1, k=0.5):
    smooth_inpI = cv2.GaussianBlur(inpI, ksize, sigma)
    outpI = (inpI - smooth_inpI)*k + inpI
    return outpI

def process_teeth(inpI, inpS, pxyF, pxyS, rsize, boundary, alpha=14, betaU=3, betaL=5, dissU=-1, dissL=0):
    regionU, regionL = detect_region(inpI, inpS, rsize, boundary, alpha, betaU, betaL)
    # enhance upper region
    tmpI  = local_enhancement(inpI, inpS, pxyF['upper'], pxyS['upper'], regionU, rsize, boundary, 'upper', dissU)
    # enhance lower region
    tmpI = local_enhancement(tmpI, inpS, pxyF['lower'], pxyS['lower'], regionL, rsize, boundary, 'lower', dissL)
    # sharpening
#    outpI = sharpen(tmpI, k=0.5)
    outpI = tmpI
    return outpI 

def test1():
    from candidate import weighted_median
    sq = Square(0.25, 0.75, 0.6, 1.00)
    tgtdata = np.load('target/target001.npz')
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
    inpdata = np.load('input/test036_ldmks.npy')
    nfr = inpdata.shape[0]
    pxyF, pxyS = process_proxy(213, 235, 0.005, 0.003, 300)
    for cnt, inpS in enumerate(inpdata):
        print("%04d/%04d" % (cnt+1, nfr))
        tmpI, tmpS = weighted_median(inpS, tgtS, tgtI, 50)
        outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, 300, boundary)
        cv2.imshow('raw', tmpI)
        cv2.imshow('enhance', outpI)
        cv2.waitKey(0)

def test2():
    # load proxy landmarks and filters
    pxyF, _ = process_proxy(213, 235, 0.005, 0.003, rsize=300)
    pxyFU, pxyFL = pxyF['upper'], pxyF['lower']
    maskU = (pxyFU > 0.5).astype(np.uint8)
    maskU[maskU != 0] = 255
    maskL = (pxyFL > 0.5).astype(np.uint8)
    maskL[maskL != 0] = 255
    cv2.imshow('Upper', maskU)
    cv2.imshow('Lower', maskL)
    cv2.waitKey(0)
    cv2.imwrite('reference/FilterUpper.png', maskU)
    cv2.imwrite('reference/FilterLower.png', maskL)
    
def test3():
    tar_path = 'target/target001.mp4'
    fr = 3261
    mode = 'upper'
    select_proxy(tar_path, mode, fr)
    
def test4(k=0.5):
    imgfiles = os.listdir('tmp/')
    for f in imgfiles:
        inpI = cv2.imread(f)
        outpI = sharpen(inpI, k=k)
        cv2.imshow('input', inpI)
        cv2.imshow('output', outpI)
        cv2.waitKey(0)
    
if __name__ == '__main__':
    k = 0.5
    test4(k)