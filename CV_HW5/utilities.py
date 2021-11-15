import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
import matplotlib
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

label_type = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
              'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

def get_data(gray=True,size=None, normal=False):

    train_x = []
    test_x = []

    train_y = []
    test_y = []

    size = size

    for index, label in enumerate(label_type):
        training_imgs = glob.glob('hw5_data/train/{}/*.jpg'.format(label))
        testing_imgs = glob.glob('hw5_data/test/{}/*.jpg'.format(label))
        for fname in training_imgs:
            train_y.append(index)
            img = cv2.imread(fname)
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if size!=None:
               	    img = cv2.resize(img, (size,size)).reshape((size,size,1))/255.0
            elif size!=None:
                img = cv2.resize(img, (size,size)).reshape((size,size,3))/255.0
            if normal:
                img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            train_x.append(img)
            
        for fname in testing_imgs:
            test_y.append(index)
            img = cv2.imread(fname)
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if size!=None:
                    img = cv2.resize(img, (size,size)).reshape((size,size,1))/255.0
            elif size!=None:
                img = cv2.resize(img, (size,size)).reshape((size,size,3))/255.0
            if normal:
                img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            test_x.append(img)
    if size!=None:
        train_x = np.array(train_x).astype(np.float32)
        test_x = np.array(test_x).astype(np.float32)
    train_y = np.array(train_y).astype(np.float32)
    test_y = np.array(test_y).astype(np.float32)
    
    return(train_x,train_y,test_x,test_y)

def plot_heatmap(true_y, pred_y, save_dir):
    true_y = [label_type[x] for x in true_y]
    pred_y = [label_type[x] for x in pred_y]
    sns.heatmap(confusion_matrix(true_y, pred_y, labels=label_type, normalize='true'),xticklabels=label_type,yticklabels=label_type)
    plt.tight_layout()
    plt.savefig(save_dir)

def plot_res(true_y, pred_y, save_dir='res'):
    
    def unique_by_key(elements, key=None):
        if key is None:
            # no key: the whole element must be unique
            key = lambda e: e
        return list({key(el): el for el in elements}.values())

    true_y = [label_type[x] for x in true_y]
    pred_y = [label_type[x] for x in pred_y]

    train = []
    test = []

    train_dict = {}
    test_dict = {}

    for index, label in enumerate(label_type):
        training_imgs = glob.glob('hw5_data/train/{}/*.jpg'.format(label))
        testing_imgs = glob.glob('hw5_data/test/{}/*.jpg'.format(label))
        for fname in training_imgs:
            img = cv2.imread(fname)
            train.append(img)
            if label not in train_dict:
                train_dict[label] = img

        for fname in testing_imgs:
            img = cv2.imread(fname)
            test.append(img)
            if label not in test_dict:
                test_dict[label] = img
                
    false_negative = {k:[] for k in label_type}
    false_positive = {k:[] for k in label_type}
    true_positive = {k:[] for k in label_type} 
    
    for idx in range(len(true_y)):
        if true_y[idx] != pred_y[idx]:
            false_negative[true_y[idx]].append((idx,pred_y[idx]))
            false_positive[pred_y[idx]].append((idx,true_y[idx]))   
        else:
            true_positive[true_y[idx]].append(idx)
            
    for cat in false_negative:
        false_negative[cat]=unique_by_key(false_negative[cat], key=itemgetter(1))

    for cat in false_positive:
        false_negative[cat]=unique_by_key(false_negative[cat], key=itemgetter(1))
        
    fig, axes = plt.subplots(nrows=16, ncols=5, figsize=(12, 30))

    axes[0][0].axis('off')
    
    for idx, cat in enumerate(label_type):
        
        axes[idx+1][1].axis('off')
        axes[idx+1][1].imshow(train_dict[cat])
        
        axes[idx+1][2].axis('off')
        if len(true_positive[cat])!=0:
            axes[idx+1][2].imshow(test[true_positive[cat][0]])
        
        axes[idx+1][3].axis('off')
        if len(false_positive[cat])!=0:
            axes[idx+1][3].set_title(false_positive[cat][0][1])
            axes[idx+1][3].imshow(test[false_positive[cat][0][0]])
            
        axes[idx+1][4].axis('off')
        axes[idx+1][4].patch.set_facecolor('xkcd:mint green')
        if len(false_negative[cat])!=0:
            axes[idx+1][4].set_title(false_negative[cat][0][1])
            axes[idx+1][4].imshow(test[false_negative[cat][0][0]])
            
    for ax, row in zip(axes[1:,0], label_type):
        ax.axis('off')
        ax.set_title(row, rotation=0, size='large',fontweight='bold',loc='right')
        
    for ax, col in zip(axes[0][1:], ["Sample training images","Sample true positives","False positives with \ntrue label",'False negatives with \nwrong predicted label']):
        ax.axis('off')
        ax.set_title(col, rotation=0, size='large',fontweight='bold',y=-0.01)
    
    fig.tight_layout()
    plt.savefig(save_dir)
    plt.show()
