# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
import numpy as np
from subprocess import call
from __init__ import inp_dir, tar_dir, outp_dir, Square
from candidate import preprocess, weighted_median
from teeth import process_proxy, process_teeth

fps = 30
size = (1280, 720)

def combine(mp4_path, mp3_path):
    bdir, namext = os.path.split(mp4_path)
    name, _ = os.path.splitext(namext)
    outp_path = bdir + '/' + name + '.mp4'
    command = 'ffmpeg -i ' + mp4_path + ' -i ' + mp3_path + ' -c:v copy -c:a aac -strict experimental ' + outp_path
    call(command)
    return outp_path

def lowerface(mp3_path, inp_path, tar_path, sq, preproc=False, n=50, rsize=300, startfr=300, endfr=None):
    # preprocess target video
    inp_dir, filename = os.path.split(inp_path)
    inp_id, _ = os.path.splitext(filename)
    tar_dir, filename = os.path.split(tar_path)
    tar_id, _ = os.path.splitext(filename)
    tmp_path = tar_dir+'/'+tar_id+'.npz'  
    
    # preprocess
    if preproc or os.path.exists(tmp_path) == False:
        preprocess(tar_path, tmp_path, rsize, startfr, endfr)
    
    # load target data  
    tgtdata = np.load(tmp_path)
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    
    # clip target textures
    boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
    
    # load input data
    inpdata = np.load(inp_path)
    nfr = inpdata.shape[0]
    
    # load proxy landmarks and filters
    pxyF, pxyS = process_proxy(213, 235, 0.005, 0.003, rsize=rsize)
    
    # create every frame and form a mp4
    print('Start to create new video...')
    avi_path = outp_dir+inp_id+'-x-'+tar_id+'.avi'
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for cnt, inpS in enumerate(inpdata):
        print("%04d/%04d" % (cnt+1, nfr))
        tmpI, tmpS = weighted_median(inpS, tgtS, tgtI, n)
        outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, boundary)
        
#        cv2.imshow('raw', tmpI)
#        cv2.imshow('enhance', outpI)
#        cv2.waitKey(0)
 
        H, W, _ = outpI.shape
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        upper, left = (360, 560)
        frame[upper:upper+H, left:left+W, :] = outpI
        writer.write(frame)
        
    outp_path = combine(avi_path, mp3_path)
    return outp_path

if __name__ == '__main__':
    inp_id  = "test036"
    tar_id  = "target001"
    sq = Square(0.25, 0.75, 0.6, 1.00)
    preproc = False
    n       = 50
    rsize   = 300
    startfr = 300
    endfr   = 4820
    
    mp3_path  = inp_dir + inp_id + ".mp3"
    inp_path  = inp_dir + inp_id + "_ldmks.npy"
    tar_path  = tar_dir + tar_id + ".mp4"
    outp_path = lowerface(mp3_path, inp_path, tar_path, sq, preproc, n, rsize, startfr, endfr)
    print('Lower face synthesized at path %s' % outp_path)
    