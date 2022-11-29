import cv2
import os
import numpy as np


train_data = []
train_labels = []
test_data = []
test_labels = []

for root,dirs,files in os.walk('BelgiumTSC_Training/Training/'):
    for name in files:
        path = os.path.join(root,name)
        if path.endswith('.ppm'):

            label = int(path.split("/")[2])
            train_labels.append(label)

            img = cv2.imread(path)
            img_reshape = cv2.resize(img,(32,32))
            train_data.append(img_reshape)


for root,dirs,files in os.walk('BelgiumTSC_Testing/Testing/'):
    for name in files:
        path = os.path.join(root,name)
        if path.endswith('.ppm'):

            label = int(path.split("/")[2])
            test_labels.append(label)

            img = cv2.imread(path)
            img_reshape = cv2.resize(img,(32,32))
            test_data.append(img_reshape)


train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

np.save('train_data.npy',train_data)
np.save('train_labels.npy',train_labels)
np.save('test_data.npy',test_data)
np.save('test_labels.npy',test_labels)