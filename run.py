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

def lowerface(sq, mp3_path, inp_path, tar_path, avi_path=None, rsize=300, preproc=False):
    # preprocess target video
    inp_dir, filename = os.path.split(inp_path)
    inp_id, _ = os.path.splitext(filename)
    tar_dir, filename = os.path.split(tar_path)
    tar_id, _ = os.path.splitext(filename)
    tmp_path = tar_dir+'/'+tar_id+'.npz'  
    
    # preprocess
    if preproc or os.path.exists(tmp_path) == False:
        preprocess(tar_path, tmp_path, rsize)
    
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
    pxyF, pxyS = process_proxy(rsize)
    
    # create every frame and form a mp4
    avi_path = '%s%s-x-%s.avi' % (outp_dir, inp_id, tar_id, ) if avi_path is None else avi_path
    print('Start to create new video...')
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for cnt, inpS in enumerate(inpdata):
        print("%s: %04d/%04d" % (avi_path, cnt+1, nfr))
        tmpI, tmpS = weighted_median(inpS, tgtS, tgtI)
        outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, boundary)
 
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
    rsize   = 300
    
    mp3_path  = inp_dir + inp_id + ".mp3"
    inp_path  = inp_dir + inp_id + "_ldmks.npy"
    tar_path  = tar_dir + tar_id + ".mp4"
    avi_path  = 'output/2teeth.avi'
    outp_path = lowerface(sq, mp3_path, inp_path, tar_path, avi_path, rsize, preproc)
    print('Lower face synthesized at path %s' % outp_path)
    