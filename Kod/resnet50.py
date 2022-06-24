from tensorflow.keras.applications import ResNet50
import numpy as np
import pandas as pd
import random
import os
import cv2
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense, Flatten




img_size = 96

def extract_label(img_path,train = True):
  filename, _ = os.path.splitext(os.path.basename(img_path))

  subject_id, etc = filename.split('__')
  
  if train:
      gender, lr, finger, _, _ = etc.split('_')
  else:
      gender, lr, finger, _ = etc.split('_')
  
  gender = 0 if gender == 'M' else 1

  return np.array([gender], dtype=np.uint16)


def loading_data(path,train):
    print("loading data from: ",path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            label = extract_label(os.path.join(path, img),train)
            data.append([label[0], img_resize ])
        except Exception as e:
            pass
    data
    return data



Real_path = "SOCOFing/Real"
Easy_path = "SOCOFing/Altered/Altered-Easy"
Medium_path = "SOCOFing/Altered/Altered-Medium"
Hard_path = "SOCOFing/Altered/Altered-Hard"

Easy_data = loading_data(Easy_path, train = True)
Medium_data = loading_data(Medium_path, train = True)
Hard_data = loading_data(Hard_path, train = True)
test = loading_data(Real_path, train = False)

data = np.concatenate([Easy_data, Medium_data, Hard_data], axis=0)

del Easy_data, Medium_data, Hard_data



random.shuffle(data)
random.shuffle(test)


data




img, labels = [], []
for label, feature in data:
    labels.append(label)
    img.append(feature)
train_data = np.array(img).reshape(-1, img_size, img_size, 1)
train_data = train_data / 255.0

train_labels = to_categorical(labels, num_classes = 2)
train_labels = to_categorical(labels, num_classes = 2)


resnet = Sequential()
pretrained = ResNet50(include_top = False, classes = 2, weights = None, input_shape=[96, 96, 1])

resnet.add(pretrained)

resnet.add(Flatten())
resnet.add(Dense(256, activation = 'relu'))
resnet.add(Dense(128, activation = 'relu'))
resnet.add(Dense(2, activation = 'sigmoid'))

resnet.summary()


resnet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = resnet.fit(train_data, train_labels, batch_size = 128, epochs = 10, 
          validation_split = 0.2)


pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)


test_images, test_labels = [], []

for label, feature in test:
    test_images.append(feature)
    test_labels.append(label)
    
test_images_resize = np.array(test_images).reshape(-1, img_size, img_size, 1)
test_images_resize = test_images_resize / 255.0
del test
test_labels  = to_categorical(test_labels, num_classes = 2)
test_images


resnet.save('resnet50_model/')


res = resnet.evaluate(test_images_resize, test_labels)
print("    Test Loss: {:.5f}".format(res[0]))
print("Test Accuracy: {:.2f}%".format(res[1] * 100))
pred = resnet.predict(test_images_resize)
pred = np.argmax(pred,axis=1)



plt.figure(figsize=(15, 15))
n_img = (4, 4)

for i in range(1, (n_img[0] * n_img[1])+1):
    plt.subplot(n_img[0], n_img[1], i)
    plt.axis('off')
    
    color='green'
    if test_labels[i][1] != pred[i]:
        color='red'
    test_title = 'F'
    if test_labels[i][1] == 0:
        test_title = 'M'
    pred_title = 'F'
    if pred[i] == 0:
        pred_title = 'M'
        
    plt.title(f"True: {test_title}\nPredicted: {pred_title}", color=color)
    plt.imshow(test_images[i], cmap='gray')


