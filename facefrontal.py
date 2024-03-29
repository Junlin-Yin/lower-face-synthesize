'''
This is the python code as a re-implementation of the matlab code from:
http://www.openu.ac.il/home/hassner/projects/frontalize/
Tal Hassner, Shai Harel*, Eran Paz* and Roee Enbar, Effective Face Frontalization in Unconstrained Images, IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
The algorithm credit belongs to them. 
I implement it as I dislike reading matlab code--i started using matlab when it was around 1GB but look at it now.....
In order to make the code run you need: 
1. compile the dlib python code: http://dlib.net/
2. download the shape_predictor_68_face_landmarks.dat file from:
http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
3. install python dependencies 
Contact me if you have problem using this code: 
heng.yang@cl.cam.ac.uk 
'''

import scipy.misc as sm
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import dlib
import pickle as pkl
from scipy import ndimage
import copy 

def plot3d(p3ds):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p3ds[:,0],p3ds[:,1], p3ds[:,2])
    plt.show()


class frontalizer():
    def __init__(self,refname):
        # initialise the model with the 3d reference face 
        # and the camera intrinsic parameters 
        # they are stored in the ref3d.pkl  
        with open(refname, 'rb') as f:
            ref = pkl.load(f, encoding='iso-8859-1')
            self.refU  = ref['refU']
            self.A = ref['outA']
            self.refxy = ref['ref_XY']
            self.p3d = ref['p3d']
            self.refimg = ref['refimg']
    def get_headpose(self,p2d):
        assert(len(p2d) == len(self.p3d))
        p3_ = np.reshape(self.p3d,(-1,3,1)).astype(np.float)
        p2_ = np.reshape(p2d,(-1,2,1)).astype(np.float)
        # print(self.A)                 # camera intrinsic matrix
        distCoeffs = np.zeros((5,1))    # distortion coefficients
        succ,rvec,tvec = cv2.solvePnP(p3_,p2_, self.A, distCoeffs)
        # rvec.shape = (3, 1) which is a compact way to represent a rotation
        # tvec.shape = (3, 1) which is used to represent a transformation
        if not succ:
            print('There is something wrong, please check.')
            return None
        else:
            matx = cv2.Rodrigues(rvec)      # matx[0] := R matrix
            ProjM_ = self.A.dot(np.insert(matx[0],3,tvec.T,axis=1))     # intrinsic * extrinsic
            return rvec,tvec,ProjM_
    def frontalization(self,img_,facebb,p2d_):
        #we rescale the face region (twice as big as the detected face) before applying frontalisation
        #the rescaled size is WD x HT 
        WD = 250 
        HT = 250 
        ACC_CONST = 800
        facebb = [facebb.left(), facebb.top(), facebb.width(), facebb.height()]
        w = facebb[2]
        h = facebb[3]
        fb_ = np.clip([[facebb[0] - w, facebb[1] - h],[facebb[0] + 2 * w, facebb[1] + 2 * h]], [0,0], [img_.shape[1], img_.shape[0]])  
        img = img_[fb_[0][1]:fb_[1][1], fb_[0][0]:fb_[1][0],:]      # img.shape <= (3h, 3w) where [w, h] is the size of face detector
        p2d = copy.deepcopy(p2d_) 
        p2d[:,0] = (p2d_[:,0] - fb_[0][0]) * float(WD) / float(img.shape[1])
        p2d[:,1] = (p2d_[:,1] - fb_[0][1]) * float(HT) / float(img.shape[0])
        img = cv2.resize(img, (WD,HT)) 
        #finished rescaling
       
        tem3d = np.reshape(self.refU,(-1,3),order='F')
        bgids = tem3d[:,1] < 0# excluding background 3d points 
        # plot3d(tem3d)
        # print tem3d.shape 
        ref3dface = np.insert(tem3d, 3, np.ones(len(tem3d)),axis=1).T   # homogeneous coordinates
        _, _, ProjM = self.get_headpose(p2d)
        proj3d = ProjM.dot(ref3dface)
        proj3d[0] /= proj3d[2]      # homogeneous normalization
        proj3d[1] /= proj3d[2]      # homogeneous normalization
        proj2dtmp = proj3d[0:2]
        #The 3D reference is projected to the 2D region by the estimated pose 
        #Then check the projection lies in the image or not 
        vlids = np.logical_and(np.logical_and(proj2dtmp[0] > 0, proj2dtmp[1] > 0), 
                               np.logical_and(proj2dtmp[0] < img.shape[1] - 1,  proj2dtmp[1] < img.shape[0] - 1))
        vlids = np.logical_and(vlids, bgids)
        proj2d_valid = proj2dtmp[:,vlids]       # totally vlids points can be projected into the query image

        sp_  = self.refU.shape[0:2]
        synth_front = np.zeros(sp_,np.float)    # 320 * 320
        inds = np.ravel_multi_index(np.round(proj2d_valid).astype(int),(img.shape[1], img.shape[0]),order = 'F')
        unqeles, unqinds, inverids, conts  = np.unique(inds, return_index=True, return_inverse=True, return_counts=True)
        tmp_ = synth_front.flatten()
        tmp_[vlids] = conts[inverids].astype(np.float)
        synth_front = tmp_.reshape(synth_front.shape,order='F')
        synth_front = cv2.GaussianBlur(synth_front, (17,17), 30).astype(np.float)

        # color all the valid projected 2d points according to the query image
        rawfrontal = np.zeros((self.refU.shape[0],self.refU.shape[1], 3)) 
        for k in range(3):
            z = img[:,:,k]
            intervalues = ndimage.map_coordinates(img[:,:,k].T,proj2d_valid,order=3,mode='nearest')
            tmp_  = rawfrontal[:,:,k].flatten()
            tmp_[vlids] = intervalues
            rawfrontal[:,:,k] = tmp_.reshape(self.refU.shape[0:2],order='F')

        mline = synth_front.shape[1]//2
        sumleft = np.sum(synth_front[:,0:mline])
        sumright = np.sum(synth_front[:,mline:])
        sum_diff = sumleft - sumright
        # print(sum_diff)
        if np.abs(sum_diff) > ACC_CONST:
            weights = np.zeros(sp_)
            if sum_diff > ACC_CONST:        # sumleft > sumright => face to left
                weights[:,mline:] = 1.
            else:                           # sumright > sumleft => face to right
                weights[:,0:mline] = 1.
            weights = cv2.GaussianBlur(weights, (33,33), 60.5).astype(np.float)
            synth_front /= np.max(synth_front) 
            weight_take_from_org = 1 / np.exp(1 + synth_front)
            weight_take_from_sym = 1 - weight_take_from_org
            weight_take_from_org = weight_take_from_org * np.fliplr(weights)
            weight_take_from_sym = weight_take_from_sym * np.fliplr(weights) 

            weights = np.tile(weights,(1,3)).reshape((weights.shape[0],weights.shape[1],3),order='F')
            weight_take_from_org = np.tile(weight_take_from_org,(1,3)).reshape((weight_take_from_org.shape[0],weight_take_from_org.shape[1],3),order='F')
            weight_take_from_sym = np.tile(weight_take_from_sym,(1,3)).reshape((weight_take_from_sym.shape[0],weight_take_from_sym.shape[1],3),order='F')
            
            denominator = weights + weight_take_from_org + weight_take_from_sym
            frontal_sym = (rawfrontal * weights + rawfrontal * weight_take_from_org + np.fliplr(rawfrontal) * weight_take_from_sym) / denominator
        else:
            frontal_sym = rawfrontal
        return rawfrontal, frontal_sym

fronter = frontalizer('reference/ref3d.pkl')
def facefrontal(img, detector, predictor):
    '''
    ### parameters
    img: original image to be frontalized \\
    detector: face detector generated by dlib.get_frontal_face_detector() \\
    predictor: landmark extractor generated by dlib.shape_predictor(...)
    ### retval
    newimg: (320, 320, 3), frontalized image
    '''
    dets = detector(img, 1)    # only 0 or 1 face in each frame
    if(len(dets) == 0):
        return None
    det = dets[0]
    shape = predictor(img, det)
    p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)
    rawfront, symfront = fronter.frontalization(img, det, p2d)
    newimg = symfront.astype('uint8')
    # cv2.imshow('newimg', newimg)
    # cv2.waitKey(0)
    return newimg

if __name__ == "__main__":
    fronter = frontalizer('reference/ref3d.pkl')
    ref3d = np.reshape(fronter.refU,(-1,3),order='F')
    plot3d(ref3d)