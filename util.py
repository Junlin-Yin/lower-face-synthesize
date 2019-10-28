# -*- coding: utf-8 -*-

import cv2
import numpy as np
from __init__ import detector, predictor
from facefrontal import facefrontal

k = 1.8
data = np.load('reference/ldmk_stat.npz')
mean, std = data['mean'], data['std']
boundL = mean - k*std
boundU = mean + k*std

def util1(mp4_path, startfr=300, endfr=None):
    # eliminate frames spoiled by facefrontal()
    textures  = []
    landmarks = []
    frames = []
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
        if np.sum(np.logical_or(ldmk < boundL, ldmk > boundU)) > 0:
            continue

        frames.append(cnt-1)        
        landmarks.append(ldmk.flatten())
        txtr = img[origin[1]:origin[1]+size[1], origin[0]:origin[0]+size[0]]
        textures.append(txtr)
        cv2.imwrite('tmp/%04d.png'%(cnt-1), txtr)
    
    landmarks = np.array(landmarks)
    textures  = np.array(textures)
    frames = np.array(frames)[1:-1]
    approx = (landmarks[2:, :] + landmarks[:-2, :]) / 2
    L2 = np.linalg.norm(landmarks[1:-1, :]-approx, ord=2, axis=1)
    landmarks = landmarks[1:-1]
    textures  = textures[1:-1]
    
    check = (L2 <= 0.1).nonzero()
    landmarks = landmarks[check]
    frames    = frames[check]
    L2        = L2[check]
    
    with open('L2.txt', 'a') as f:
        for i, loss in zip(frames, L2):
            f.write('Frame %04d: %.3f\n' % (i, loss))
    print('Done')
    
def util2(mp4_path, save_path, startfr=0, endfr=None):
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
        _, img = cap.read()
        img = img[360:360+120, 560:560+150]
        cv2.imwrite('%s%04d.png' % (save_path, cnt-1), img)
    print('Done')
            
if __name__ == '__main__':
    mp4_path = 'C:/Users/xinzhu/Desktop/lower-face-synthesize/3prefilter.mp4'
    save_path = 'tmp/'
    util2(mp4_path, save_path)