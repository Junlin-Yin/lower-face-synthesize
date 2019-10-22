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
teeth_lower = np.array([0, 0, 200])
teeth_upper = np.array([180, 50, 255])

leftidxlist  = np.array([0, 12])
rightidxlist = np.array([6, 16])
upperidxlist = np.array([0, 6, 12, 13, 14, 15, 16])
loweridxlist = np.array([0, 6, 12, 16, 17, 18, 19])

class Square:
    def __init__(self, l, r, u, d):
        self.left = l
        self.right = r
        self.up = u
        self.down = d
        
    def align(self, S):
        left  = round(self.left  * S)
        right = round(self.right * S)
        upper = round(self.up    * S)
        lower = round(self.down  * S)
        return left, right, upper, lower

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
        landmarks.append(ldmk)
        
        # resize texture into a square
        txtr = img[origin[1]:origin[1]+size[1], origin[0]:origin[0]+size[0]]
        txtr = cv2.resize(txtr, (rsize, rsize))       
        # mask & inpaint for clothes region
        txtr = mask_inpaint(txtr)
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

def teeth_proxy():  
    # select and generate teeth proxy frame(s)
    startfr = 78 * 30
    rsize   = 300
    tar_path = 'target/target001.mp4'
    cap = cv2.VideoCapture(tar_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
    while True:
        ret, img = cap.read()
        cv2.imshow('', img)
        cv2.waitKey(0)
        a = input('Frame %04d: OK?(y/n)' % startfr)
        if a == 'y':
            break
        startfr += 1

    det = detector(img, 1)[0]
    origin = np.array([det.left(), det.top()])
    size = np.array([det.width(), det.height()])
    txtr = img[origin[1]:origin[1]+size[1], origin[0]:origin[0]+size[0]]
    txtr = cv2.resize(txtr, (rsize, rsize))
    cv2.imshow('', txtr)
    cv2.waitKey(0)
    cv2.imwrite('reference/proxy_lower.png', txtr)

def teeth_region(inpI, inpS, rsize, boundary):
    # automatically detect upper and lower teeth region in input image
    # boundary: (left, right, upper, lower)
    cv2.imshow('raw',  inpI)
    cv2.waitKey(0)
    
    hsv = cv2.cvtColor(inpI, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, teeth_lower, teeth_upper)
    xs, ys = mask.nonzero()
    region = np.array([[x, y] for x, y in zip(xs, ys)])
    if region.shape == (0,):
        raise Exception("No teeth region found!")
    
    # eliminate region which is out of lip 
    left  = np.min(inpS[leftidxlist,  0])
    right = np.max(inpS[rightidxlist, 0])
    upper = np.min(inpS[upperidxlist, 1])
    lower = np.max(inpS[loweridxlist, 1])
    
    left  = round((left  - boundary[0]) * rsize)
    right = round((right - boundary[0]) * rsize)
    upper = round((upper - boundary[2]) * rsize)
    lower = round((lower - boundary[2]) * rsize)

    check = np.logical_and(
            np.logical_and(region[:, 0] >= left,  region[:, 0] <= right),
            np.logical_and(region[:, 1] >= upper, region[:, 1] <= lower))
    region = region[check.nonzero()]
    
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    print(region.shape)
    return region
    
def teeth_enhancement(inpI, inpS, pxyU, pxyL):
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
    # load target data  
    tgtdata = np.load('target/target001.npz')
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    
    # clip target textures
    sq = Square(0.25, 0.75, 0.6, 1.00)
    left, right, upper, lower = sq.align(tgtI.shape[1])
    tgtI = tgtI[:, upper:lower, left:right, :]
    
    # load input data
    inpdata = np.load('input/test036_ldmks.npy')
    
    # create every frame and form a mp4
    for cnt, inpS in enumerate(inpdata):
        outpI, outpS = weighted_median(inpS, tgtS, tgtI, 50)
        region = teeth_region(outpI, outpS, 300, (left, right, upper, lower))
