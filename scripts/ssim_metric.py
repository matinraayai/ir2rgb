#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:47:23 2020

@author: johndoe
"""

#ssim_test

import numpy as np

from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray

def mse_single_frame(original, predicted):
    original_gray = rgb2gray(original)
    predicted_gray = rgb2gray(predicted)
    
    return np.linalg.norm(original_gray - predicted_gray)

def mse_movie(original, predicted):
    # shape needs to be [num_frames, height, width]
    assert original.shape == predicted.shape # need same number frames and shape
    
    ssim_total = 0
    for frame in original.shape[0]:
        ssim_total += mse_single_frame(original[frame, :,:], predicted[frame, :,:])
        
    return(ssim_total/original.shape[0])

def ssim_single_frame(original, predicted):
    original_g = rgb2gray(original)
    predicted_g = rgb2gray(predicted)
    
    return(ssim(original_g, predicted_g, dynamic_range=original_g.max() - predicted_g.min()))
    
def ssim_movie(original, predicted):
    # shape needs to be [num_frames, height, width]
    assert original.shape == predicted.shape # need same number frames and shape
    
    ssim_total = 0
    for frame in original.shape[0]:
        ssim_total += ssim_single_frame(original[frame, :,:], predicted[frame, :,:])
        
    return(ssim_total/original.shape[0])

