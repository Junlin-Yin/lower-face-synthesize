# -*- coding: utf-8 -*-

import cv2
import numpy as np

k = 1.8
data = np.load('reference/ldmk_stat.npz')
mean, std = data['mean'], data['std']
boundL = mean - k*std
boundU = mean + k*std

def util1(mp4_path, startfr=530, endfr=None):
    # eliminate frames spoiled by facefrontal()
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
        cv2.imshow('', img)
        cv2.waitKey(0)
        
    
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
    mp4_path = 'target/target001.mp4'
    util1(mp4_path)