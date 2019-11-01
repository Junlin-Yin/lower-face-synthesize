# -*- coding: utf-8 -*-

import os
import math
import cv2
import librosa
import numpy as np
from subprocess import call

vfps = 30
size = (1280, 720)

def align2audio(data, mp3_path):
    n = 100
    wave, sr = librosa.load(mp3_path, sr=vfps*n)
    Nfr = math.ceil(wave.shape[0] / n)
    line = data[-1]
    tail = np.array([line]*(Nfr-data.shape[0]))
    newdata = np.concatenate([data, tail], axis=0)
    return newdata

def combine(mp4_path, mp3_path):
    bdir, namext = os.path.split(mp4_path)
    name, _ = os.path.splitext(namext)
    outp_path = bdir + '/' + name + '.mp4'
    command = 'ffmpeg -i ' + mp4_path + ' -i ' + mp3_path + ' -c:v copy -c:a aac -strict experimental ' + outp_path
    call(command)
    return outp_path

def formMp4(res_path, mp3_path, avi_path=None, fps=vfps, size=size):
    # preprocess target video
    res_dir, filename = os.path.split(res_path)
    res_id, _ = os.path.splitext(filename) 
    avi_path = '%s/%s.avi' % (res_dir, res_id) if avi_path is None else avi_path
    
    imgdata = np.load(res_path)
    print('Start to create new video...')
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for cnt, img in enumerate(imgdata):
        H, W, _ = img.shape
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        upper, left = (360, 560)
        frame[upper:upper+H, left:left+W, :] = img
        writer.write(frame)
        
    video_path = combine(avi_path, mp3_path)
    return video_path

if __name__ == '__main__':
    formMp4('output/i36t1.npy', 'input/test036.mp3', 'output/i36t1.avi')