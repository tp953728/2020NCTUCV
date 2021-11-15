#/usr/bin/env python
# coding: utf-8

# # Bag of SIFT representation + nearest neighbor classifier

# In[2]:

from __future__ import print_function
import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import glob
from scipy.cluster.vq import *
import matplotlib.pyplot as plt
import time
from utilities import get_data, plot_heatmap, plot_res

label_type = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'InsideCity', 'Kitchen', 'LivingRoom', 'Mountain', 'Office','OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding']
save_dir = "result/Task2"


# Detect train data feature

des_list = []
path = "hw5_data/train/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    #print(File)
    im = cv2.imread(File)
    #print(im)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX)
    #im = cv2.resize(im, (200,200), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    des_list.append((File, des1))
    #print(len(des_list))
    
des_list_0 = des_list[0]
descriptors = des_list_0[1]

for image_path, descriptor in des_list[1:]:
    if descriptor is None:
        print(0)
        continue
    #print(descriptor.shape)
    descriptors = np.vstack((descriptors, descriptor))
    #print(descriptor.shape)
#print(descriptors)
t2 = time.time()
print("Detect train data time: ", t2-t1)

#print(descriptors.shape)


# Detect test data feature

test_list = []
path = "hw5_data/test/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    #im = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX)
    #im = cv2.resize(im, (200,200), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    test_list.append((File, des1))
    #print(des1)
    
t2 = time.time()
print("Detect test data time: ", t2-t1)


# K means for all feature
# Perform k-means clustering
t1 = time.time()
k = 295
voc, variance = kmeans(descriptors, k, 1) 
t2 = time.time()
print("Perform k-means clustering time: ", t2-t1)


# ## Enlarge feature number

des_list = []
path = "hw5_data/train/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    im = cv2.resize(im, (600,600), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    des_list.append((File, des1))
    #print(len(des_list))

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    if descriptor is None:
        print(0)
        continue
    #print(descriptor.shape)
    descriptors = np.vstack((descriptors, descriptor))
    #print(descriptor.shape)
#print(descriptors)

t2 = time.time()
print("Enlarge feature training data time:", t2-t1)


test_list = []
path = "hw5_data/test/**/*"

t1 = time.time()
files = glob.glob(path)
sift = cv2.xfeatures2d.SIFT_create()
for File in files:
    im = cv2.imread(File)
    im = cv2.resize(im, (1000,1000), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(im,None)
    test_list.append((File, des1))
    #print(des1)
    
t2 = time.time()
print("Enlarge feature testing data time:", t2-t1)

#Histogram of features based on K means center for each training image
# Calculate the histogram of features

im_features = np.zeros((1500, k), "float32")
for i in range(1500):
    if des_list[i][1] is None:
        continue
    words, distance = vq(des_list[i][1],voc)

    for w in words:
        im_features[i][w] += 1
    #im_features[i] /= np.sum(im_features[i])
    #im_features[i] /= np.sqrt(np.sum(im_features[i]**2))
    im_features[i] = (im_features[i] - np.mean(im_features[i])) / np.std(im_features[i])

print("img feature: ", im_features)
true_y=im_features

# Histogram of features based on K means center for each testing image
# Calculate the histogram of features

test_features = np.zeros((150, k), "float32")
for i in range(150):
    if test_list[i][1] is None:
        continue
    words, distance = vq(test_list[i][1],voc)
    
    for w in words:
        test_features[i][w] += 1
    #test_features[i] /= np.sum(test_features[i])
    #test_features[i] /= np.sqrt(np.sum(test_features[i]**2))
    test_features[i] = (test_features[i] - np.mean(test_features[i])) / np.std(test_features[i])

print("test feature: ", test_features)
pred_y=test_features

# KNN classifier

def Euclidian(a, b):
    return np.sqrt(np.sum((a-b)**2))
    #return np.linalg.norm(a-b)

def KNN(test, center, K):
    dtype = [('dis', float), ('idx', int)]
    distance = np.array([(Euclidian(test, center[i]),  i) for i in range(len(center))], dtype=dtype)
    #print (distance)
    newdistance = np.sort(distance, order='dis')
    #print (newdistance)
    
    class_count = np.zeros(15)
    for i in range(K):
        _, idx = newdistance[i]
        class_count[idx//100] += 1
        
    #print (class_count)
    #print (np.argmax(class_count))
    return np.argmax(class_count)
    
minIdx = 0
count = 0.
total = 0.
best = 0

test_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
          5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
          6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
          7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
          9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
          10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
          11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
          12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
          13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
          14, 14, 14, 14, 14, 14, 14, 14, 14, 14]

for k in range(0,100,5):
    pred = []
    total = 0.
    for i in range(15):
        count = 0.
        for j in range(10):       
            minIdx = KNN(test_features[i*10+j], im_features, k)
            pred.append(minIdx)
            if minIdx == i:
                count += 1.
        total += count
    if total>best:
        y_pred = pred
        best = total
    print(k, "total:", total/150.)
    
y_pred = np.array(y_pred)
y_true = test_y

plot_heatmap(y_true,y_pred,'hw5_data/result/knn1')
plot_res(y_true,y_pred,'hw5_data/result/knn2')