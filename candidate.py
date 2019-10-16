# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import cv2
import dlib
import os
from subprocess import call
from facefrontal import facefrontal

tar_dir = 'target/'
inp_dir = 'input/'
outp_dir= 'output/'
fps = 30
size = (1280, 720)

pdctdir = 'reference/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pdctdir)

class Square:
    def __init__(self, l, r, u, d):
        self.left = l
        self.right = r
        self.up = u
        self.down = d
        
    def align(self, origin, scale):
        pt1 = np.array([self.left, self.up]) * scale + origin
        pt2 = np.array([self.right, self.down]) * scale + origin
        pt1   = pt1.astype(np.uint)
        pt2 = pt2.astype(np.uint)
        return Square(pt1[0], pt2[0], pt1[1], pt2[1])
        
        
def preprocess(mp4_path, save_path, sq, rsize, startfr=300, endfr=None):
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
    while cap.isOpened():
        if cnt == endfr:
            break
        print(cnt)
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
        landmarks.append(ldmk)
    
        # clip the lower face image
        ssq = sq.align(origin, size)
        txtr = img[ssq.up:ssq.down, ssq.left:ssq.right]
        txtr = cv2.resize(txtr, (rsize, rsize))
#        cv2.imwrite('tmp/'+str(cnt)+'_clip.png', txtr)
        textures.append(txtr)
    
    landmarks = np.array(landmarks)
    textures = np.array(textures)
    np.savez(save_path, landmarks=landmarks, textures=textures)

def mask_inpaint(path):
    pass

def optimize_sigma(L2, n, alpha):
    left, right = 0, 1
    epsilon = 1e-2
    while True:
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
    L2 = np.linalg.norm(tgtS-inpS, ord=2, axis=(1, 2))
    weights, indices = optimize_sigma(L2, n, alpha)
    candI = tgtI[indices, :, :, :]
    
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
    return outpI            

def combine(mp4_path, mp3_path):
    bdir, namext = os.path.split(mp4_path)
    name, _ = os.path.splitext(namext)
    outp_path = bdir + '/' + name + '.mp4'
    command = 'ffmpeg -i ' + mp4_path + ' -i ' + mp3_path + ' -c:v copy -c:a aac -strict experimental ' + outp_path
    call(command)
    os.remove(mp4_path)
    return outp_path

def lowerface(mp3_path, inp_path, tar_path, preproc=False, n=50, rsize=150, startfr=300, endfr=None):
    # preprocess target video
    inp_dir, filename = os.path.split(inp_path)
    inp_id, _ = os.path.splitext(filename)
    tar_dir, filename = os.path.split(tar_path)
    tar_id, _ = os.path.splitext(filename)
    tmp_path = tar_dir+tar_id+'.npz'  
    if preproc or os.path.exists(tmp_path) == False:
        sq = Square(0.2, 0.8, 0.40, 1.00)
        preprocess(tar_path, tmp_path, sq, rsize, startfr, endfr)
        
    tgtdata = np.load(tmp_path)
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    inpdata = np.load(inp_path)
    
    avi_path = outp_dir+inp_id+' x '+tar_id+'.avi'
    writer = cv2.VideoWriter(avi_path+inp_id+' x '+tar_id+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for cnt, inpS in enumerate(inpdata):
        print(cnt)
        outpI = weighted_median(inpS, tgtS, tgtI, n)
        
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        upper, left = (360, 560)
        frame[upper:upper+rsize, left:left+rsize, :] = outpI
        writer.write(frame)
        
    return combine(avi_path, mp3_path)

sq = Square(0.2, 0.8, 0.40, 1.00)
if __name__ == '__main__':
#    tar_id = "target001"
#    savepath = preprocess(tar_dir+tar_id+'.mp4', sq, rsize=150, startfr=300)
    savepath = 'target/target001.npz'
    data = np.load(savepath)
    landmarks = data['landmarks']
    textures = data['textures']
    
    inp_id = "test036_ldmks"
    ldmks = np.load(inp_dir+inp_id+'.npy')
    ldmk_test = ldmks[160, :, :]    # the 100th frame
    outp = weighted_median(ldmk_test, landmarks, textures, n=50)
    
    cv2.imshow('title', outp)
    cv2.waitKey(0)