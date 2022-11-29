import os
import numpy as np
import imageio
from skimage import transform 

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

            img = imageio.imread(path)
            train_data.append(img)
            
train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_data = [transform.resize(image, (32, 32)) for image in train_data]
train_data = np.array(train_data)


for root,dirs,files in os.walk('BelgiumTSC_Testing/Testing/'):
    for name in files:
        path = os.path.join(root,name)
        if path.endswith('.ppm'):

            label = int(path.split("/")[2])
            test_labels.append(label)

            img = imageio.imread(path)
            test_data.append(img)



test_data = np.array(test_data)
test_labels = np.array(test_labels)

test_data = [transform.resize(image, (32, 32)) for image in test_data]
test_data = np.array(test_data)

print(train_data.shape,train_labels.shape)
print(test_data.shape,test_labels.shape)

np.save('train_data.npy',train_data)
np.save('train_labels.npy',train_labels)
np.save('test_data.npy',test_data)
np.save('test_labels.npy',test_labels)



