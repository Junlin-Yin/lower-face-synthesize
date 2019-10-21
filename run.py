# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
import numpy as np
from subprocess import call
from candidate import inp_dir, tar_dir, outp_dir
from candidate import Square, preprocess, weighted_median

fps = 30
size = (1280, 720)

def combine(mp4_path, mp3_path):
    bdir, namext = os.path.split(mp4_path)
    name, _ = os.path.splitext(namext)
    outp_path = bdir + '/' + name + '.mp4'
    command = 'ffmpeg -i ' + mp4_path + ' -i ' + mp3_path + ' -c:v copy -c:a aac -strict experimental ' + outp_path
    call(command)
    os.remove(mp4_path)
    return outp_path

def lowerface(mp3_path, inp_path, tar_path, sq, preproc=False, n=50, rsize=300, startfr=300, endfr=None):
    # preprocess target video
    inp_dir, filename = os.path.split(inp_path)
    inp_id, _ = os.path.splitext(filename)
    tar_dir, filename = os.path.split(tar_path)
    tar_id, _ = os.path.splitext(filename)
    tmp_path = tar_dir+'/'+tar_id+'.npz'  
    if preproc or os.path.exists(tmp_path) == False:
        preprocess(tar_path, tmp_path, rsize, startfr, endfr)
    
    # load target data  
    tgtdata = np.load(tmp_path)
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    
    # clip target textures
    sq = Square(0.25, 0.75, 0.55, 1.00)
    left, right, upper, lower = sq.align(tgtI.shape[1])
    tgtI = tgtI[:, upper:lower, left:right, :]
    
    # load input data
    inpdata = np.load(inp_path)
    nfr = inpdata.shape[0]
    
    # create every frame and form a mp4
    print('Start to create new video...')
    avi_path = outp_dir+inp_id+'-x-'+tar_id+'.avi'
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for cnt, inpS in enumerate(inpdata):
        print("%04d/%04d" % (cnt+1, nfr))
        outpI = weighted_median(inpS, tgtS, tgtI, n)
        
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        upper, left = (360, 560)
        frame[upper:upper+rsize, left:left+rsize, :] = outpI
        writer.write(frame)
        
    return combine(avi_path, mp3_path)

if __name__ == '__main__':
    inp_id  = "test036"
    tar_id  = "target001"
    sq = Square(0.25, 0.75, 0.55, 1.00)
    preproc = False
    n       = 50
    rsize   = 150
    startfr = 300
    endfr   = None
    
    mp3_path  = inp_dir + inp_id + ".mp3"
    inp_path  = inp_dir + inp_id + "_ldmks.npy"
    tar_path  = tar_dir + tar_id + ".mp4"
    outp_path = lowerface(mp3_path, inp_path, tar_path, sq, preproc, n, rsize, startfr, endfr)
    print('Lower face synthesized at path %s' % outp_path)
    