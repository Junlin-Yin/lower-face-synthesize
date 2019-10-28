# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import cv2
from __init__ import detector, predictor
from __init__ import inp_dir
from facefrontal import facefrontal

k = 1.8
data = np.load('reference/ldmk_stat.npz')
mean, std = data['mean'], data['std']
boundL = mean - k*std
boundU = mean + k*std

black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 255, 46])
white_lower = np.array([0, 0, 46])
white_upper = np.array([180, 50, 255])

def mask_inpaint(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, black_lower, black_upper)
    mask2 = cv2.inRange(hsv, white_lower, white_upper)
    mask = mask1 | mask2
    mimg = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
    return mimg        
        
def preprocess(mp4_path, save_path, rsize, startfr=300, endfr=None):
    '''
    ### parameters
    mp4_path: path of mp4 file \\
    sq: Squre instance which defines the boundary of lower face texture
    rsize: width (height) of clipped texture in every target video frame

    ### retval
    savepath: path that saves landmarks and textures
    '''
    landmarks = []
    textures = []
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
    cnt = startfr
    endfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) if endfr is None else endfr
    print('Start preprocessing...')
    while cap.isOpened():
        if cnt == endfr:
            break
        print("%04d/%04d" % (cnt, endfr-1))
        cnt += 1
        
        ret, img_ = cap.read()
        img = facefrontal(img_, detector, predictor)
        if img is None:
            continue
        dets = detector(img, 1)
        if len(dets) != 1:
            continue
        det = dets[0]
        shape = predictor(img, det)
        ldmk = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(48, shape.num_parts)], np.float32)
    
        # normalization according to det.shape & reshape into 40-D features
        origin = np.array([det.left(), det.top()])
        size = np.array([det.width(), det.height()])
        ldmk = (ldmk - origin) / size         # restrained in [0, 0] ~ [1, 1]
        # validate landmarks using statistics in the dataset
        if np.sum(np.logical_or(ldmk < boundL, ldmk > boundU)) > 0:
            continue        
        landmarks.append(ldmk.flatten())
        
        # resize texture into a square
        txtr = img[origin[1]:origin[1]+size[1], origin[0]:origin[0]+size[0]]
        txtr = cv2.resize(txtr, (rsize, rsize))       
        # mask & inpaint for clothes region
        txtr = mask_inpaint(txtr)
        textures.append(txtr)
        
    landmarks = np.array(landmarks)
    textures = np.array(textures)
    
    # filter frames which are not locally smooth
    approx = (landmarks[2:, :] + landmarks[:-2, :]) / 2
    L2 = np.linalg.norm(landmarks[1:-1, :]-approx, ord=2, axis=1)
    check = (L2 <= 0.1).nonzero()
    landmarks = landmarks[1:-1][check].reshape((-1, 20, 2))
    textures  = textures[1:-1][check]
    
    np.savez(save_path, landmarks=landmarks, textures=textures)

def optimize_sigma(L2, n, alpha):
    left, right = 0, 1
    epsilon = 1e-2
    i, maxI = 0, 20
    while True:
        if(i == maxI):
            raise Exception("Infinite loop in optimize_sigma()!")
        i += 1
        
        sigma = (left + right) / 2
        weights = np.exp(-L2 / (2 * sigma**2))
        indices = np.argsort(weights)[::-1] # large -> small
        weights = weights[indices]
        ratio = np.sum(weights[:n]) / np.sum(weights)
        if abs(alpha-ratio) < epsilon:
            break
        elif ratio < alpha:
            right = sigma
        else:
            left = sigma
    # ratio is highly close to alpha, meaning we find a proper sigma
    weights = weights[:n]
    weights /= np.sum(weights)              # np.sum(weights) = 1
    indices = indices[:n]
    return weights, indices        

def locate_median(weights):
    # make sure that np.sum(weights) == 1
    s = 0
    for i in range(weights.shape[0]):
        s += weights[i]
        if s >= 0.5:
            break
    return i

def weighted_median(inpS, tgtS, tgtI, n, alpha=0.9):
    # choose n candidates
    L2 = np.linalg.norm(tgtS-inpS, ord=2, axis=(1, 2))
    weights, indices = optimize_sigma(L2, n, alpha)
    candI = tgtI[indices, :, :, :]
    candS = tgtS[indices, :, :]
    
    # calculate weighted-average of landmarks for teeth enhancement
    outpS = np.sum([l*w for l, w in zip(candS, weights)], axis=0)
    
    # form output texture
    outpI = np.zeros(tgtI.shape[1:], dtype=np.uint8)
    for y in range(outpI.shape[0]):
        for x in range(outpI.shape[1]):
            # at the position (x, y)
            for c in range(outpI.shape[2]):
                # in each channel
                intencity = candI[:, y, x, c]
                indices = np.argsort(intencity)
                weights_sort = weights[indices]
                idx = locate_median(weights_sort)
                outpI[y, x, c] = intencity[indices[idx]]
    
    return outpI, outpS  

def test1():
#    tar_id = "target001"
#    savepath = preprocess(tar_dir+tar_id+'.mp4', sq, rsize=150, startfr=300)
    savepath = 'target/target001.npz'
    data = np.load(savepath)
    landmarks = data['landmarks']
    textures = data['textures']
    
    inp_id = "test036_ldmks"
    ldmks = np.load(inp_dir+inp_id+'.npy')
    ldmk_test = ldmks[160, :, :]    # the 100th frame
    outpI, outpS = weighted_median(ldmk_test, landmarks, textures, n=50)
    
    cv2.imshow('title', outpI)
    cv2.waitKey(0)
    
def test2():
    import os
    imgfiles = os.listdir('tmp/')
    for imgfile in imgfiles:
        img = cv2.imread('tmp/' + imgfile)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 46])
        white_lower = np.array([0, 0, 46])
        white_upper = np.array([180, 50, 255])
        mask1 = cv2.inRange(hsv, black_lower, black_upper)
        mask2 = cv2.inRange(hsv, white_lower, white_upper)
        mask = mask1 | mask2
        
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        
        masked = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
        
        cv2.imshow('masked', masked)
        cv2.waitKey(0)    
    
if __name__ == '__main__':
    test1()
    