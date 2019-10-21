# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import cv2
import dlib
from facefrontal import facefrontal

tar_dir = 'target/'
inp_dir = 'input/'
outp_dir= 'output/'

pdctdir = 'reference/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pdctdir)

black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 255, 46])
white_lower = np.array([0, 0, 46])
white_upper = np.array([180, 50, 255])

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

def mask_inpaint(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, black_lower, black_upper)
    mask2 = cv2.inRange(hsv, white_lower, white_upper)
    mask = mask1 | mask2
    mimg = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
    return mimg        
        
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
        landmarks.append(ldmk)
    
        # clip the lower face image
        ssq = sq.align(origin, size)
        txtr = img[ssq.up:ssq.down, ssq.left:ssq.right]
        
        # mask & inpaint for clothes region
        txtr = mask_inpaint(txtr)
        
        txtr = cv2.resize(txtr, (rsize, rsize))
#        cv2.imwrite('tmp/'+str(cnt)+'_clip.png', txtr)
        textures.append(txtr)
    
    landmarks = np.array(landmarks)
    textures = np.array(textures)
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

def teeth_proxy(img, pxy1, pxy2):
    pass

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
    outp = weighted_median(ldmk_test, landmarks, textures, n=50)
    
    cv2.imshow('title', outp)
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
    print('Hello, World')