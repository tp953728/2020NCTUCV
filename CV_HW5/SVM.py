import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/hank/libsvm-3.24/python")
from svmutil import *

from utilities import get_data, plot_heatmap, plot_res
from bag_of_features import bag_of_features

train_x, train_y, test_x, test_y = get_data(gray=False)

if os.path.isfile('vocab.pkl'):
    with open('vocab.pkl', 'rb') as handle:
    voc = pickle.load(handle)
    with open('train_features.pkl', 'rb') as handle:
        im_features = pickle.load(handle)
    with open('test_features.pkl', 'rb') as handle:
        test_features = pickle.load(handle)
else:
    im_features, test_features, voc = bag_of_features(train_x, test_x, k=400)
    with open('vocab.pkl', 'wb') as handle:
        pickle.dump(voc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('train_features.pkl', 'wb') as handle:
        pickle.dump(im_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('test_features.pkl', 'wb') as handle:
        pickle.dump(test_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

c_begin,c_end,c_step = (-5,10,1) # Cost
best_acc = 0
best_params = 0
for c in range(c_begin,c_end+c_step,c_step):
    
    # Linear kernel
    params = '-q -v 5 -s 0 -t 0 -c {}'.format(2**c)
    acc = svm_train(train_y,im_features,params)
    if acc > best_acc:
        best_acc = acc
        best_params = 2**c
    print('[Linear] {} {} (best: c={} acc={})'.format(c,acc,best_params,best_acc))

print('Kernel: linear')
params = '-q -t 0 -s 0 -c 512'
model = svm_train(train_y,im_features,params)
pred, acc, _ = svm_predict(test_y,test_features, model)
print(acc[0])

true_y = list(test_y.astype(int))
pred_y = list(map(int, pred))

plot_heatmap(true_y,pred_y,'./result/heatmap_svm')
plot_res(true_y,pred_y,'./result/res_svm')




