# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
import numpy as np
from __init__ import inp_dir, tar_dir, outp_dir, Square, detector
from candidate import preprocess, weighted_median
from teeth import process_proxy, process_teeth
from visual import align2audio, formMp4

def lowerface(sq, mp3_path, inp_path, tar_path, res_path=None, rsize=300, preproc=False):
    # preprocess target video
    inp_dir, filename = os.path.split(inp_path)
    inp_id, _ = os.path.splitext(filename)
    tar_dir, filename = os.path.split(tar_path)
    tar_id, _ = os.path.splitext(filename)
    tmp_path = tar_dir+'/'+tar_id+'.npz'  
    
    # preprocess
    if preproc or os.path.exists(tmp_path) == False:
        preprocess(tar_path, tmp_path, rsize)
    
    # load target data and clip them
    tgtdata = np.load(tmp_path)
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
    
    # load input data and proxy data
    inpdata = np.load(inp_path)
    nfr = inpdata.shape[0]
    pxyF, pxyS = process_proxy(rsize)
    
    # create every frame and form a mp4
    res_path = '%s%s-x-%s.npy' % (outp_dir, inp_id, tar_id, ) if res_path is None else res_path
    outpdata = []
    print('Start to create new video...')
    for cnt, inpS in enumerate(inpdata):
        print("%s: %04d/%04d" % (res_path, cnt+1, nfr))
        tmpI, tmpS = weighted_median(inpS, tgtS, tgtI)
        outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, boundary)
        outpdata.append(outpI)
    outpdata = np.array(outpdata)
    outpdata = align2audio(outpdata, mp3_path)
    np.save(res_path, outpdata)
    
    return res_path  
        
def util1(mp4_path, save_path, startfr=0, endfr=None):
    # extract every frame from result video
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
    cnt = startfr
    endfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) if endfr is None else endfr
    print('Start preprocessing...')
    while cap.isOpened():
        if cnt == endfr:
            break
#        print("%04d/%04d" % (cnt, endfr-1))
        cnt += 1
        _, img = cap.read()
#        img = img[360:360+120, 560:560+150]
        cv2.imwrite('%s%04d.png' % (save_path, cnt-1), img)
    print('Done')

def run():
    inp_id  = "test036"
    tar_id  = "target001"
    sq = Square(0.25, 0.75, 0.6, 1.00)
    preproc = False
    rsize   = 300
    
    mp3_path  = inp_dir + inp_id + ".mp3"
    inp_path  = inp_dir + inp_id + "_ldmks.npy"
    tar_path  = tar_dir + tar_id + ".mp4"
    res_path  = 'output/3sharp0.5.npy'
    res_path  = lowerface(sq, mp3_path, inp_path, tar_path, res_path, rsize, preproc)
    vid_path  = formMp4(res_path, mp3_path)
    print('Lower face synthesized at path %s' % vid_path)

if __name__ == '__main__':
     run()