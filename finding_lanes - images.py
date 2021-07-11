#!/usr/bin/env python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.image as mpimg
import os
from time import sleep
get_ipython().run_line_magic('matplotlib', 'inline')

os.chdir('test_images') #navigate to the directory containing sample images and load 
imglst = []
for ipimages in os.listdir():
    imglst.append(ipimages)

###Function to pre process image . Convert to Gray,apply GaussianBlur, apply canny edge detector. 
def gaussPcannyEdges(ipimage):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return(edges)

#Function to define Region of Interest.
def masked_ROI(edge_img):
    mask = np.zeros_like(edge_img)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(470, 320), (490,320), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edge_img,mask)
    return(masked_edges)

#Function to detect lines with Hough Transform
def detect_line(roi,image):
    rho = 2
    theta = np.pi/180
    threshold = 50
    min_line_length = 50
    max_line_gap = 200
    line_image = np.copy(image)*0
    lines = cv2.HoughLinesP(roi,rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    line_edges = cv2.addWeighted(image,0.8,line_image,1,0)
    return(line_edges,lines)

###Function to average the detected lines
def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope,intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit,axis=0)
        left_line = make_points(image,left_fit_average)
        right_line = make_points(image,right_fit_average)
        averaged_lines = [left_line,right_line]
        return(averaged_lines)

### function to get coordinates of extrapolated lines
def make_points(image,line):
    slope,intercept = line
    y1 = int(image.shape[0]) #bottom of the line
    y2 = int(y1*3/4.5)  #slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1,y1,x2,y2]]

#Function to overlay lane detector over original image. 
def display_lines(image,lines):
    line_image2 = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image2,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image2

img = mpimg.imread(imglst[0]) #load the desired test_image 
image = np.copy(img) 
edge_img = gaussPcannyEdges(image)
roi = masked_ROI(edge_img)
line_img,lines =detect_line(roi,image)
averaged_lines = average_slope_intercept(image,lines)
line_image = display_lines(image,averaged_lines)
final_image = cv2.addWeighted(image, 0.8, line_image,1,1)
plt.imshow(final_image)




