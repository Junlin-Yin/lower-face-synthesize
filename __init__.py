# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import dlib
pdctdir = 'reference/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pdctdir)

tar_dir = 'target/'
inp_dir = 'input/'
outp_dir= 'output/'

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