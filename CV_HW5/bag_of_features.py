import cv2
import numpy as np
from scipy.cluster.vq import *

def bag_of_features(train, test, k=300):

	des_list_train = []
	des_list_test = []

	sift = cv2.xfeatures2d.SIFT_create()

	for im in train:
	    kp, des = sift.detectAndCompute(im,None)
	    des_list_train.append(des)
	descriptors = np.concatenate(des_list_train, axis=0).astype('float32')

	for im in test:
	    kp, des = sift.detectAndCompute(im,None)
	    des_list_test.append(des)
	    
	print("Running K-means with k =",k)
	voc, _ = kmeans2(descriptors, k, minit="++")

	im_features = np.zeros((1500, k), "float32")
	for i in range(1500):
	    if des_list_train[i] is None:
	        continue
	    words, distance = vq(des_list_train[i],voc)

	    for w in words:
	        im_features[i][w] += 1
	    im_features[i] = (im_features[i] - np.mean(im_features[i])) / np.std(im_features[i])

	test_features = np.zeros((150, k), "float32")
	for i in range(150):
	    if des_list_test[i] is None:
	        continue
	    words, distance = vq(des_list_test[i],voc)
	    
	    for w in words:
	        test_features[i][w] += 1
	    test_features[i] = (test_features[i] - np.mean(test_features[i])) / np.std(test_features[i])

	return im_features, test_features, voc
