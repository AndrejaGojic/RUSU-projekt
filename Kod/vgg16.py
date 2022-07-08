from tensorflow.keras.applications import VGG16
import numpy as np
import pandas as pd
import random
import os
import cv2
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten



img_size = 96

def extract_id(img_path,real):
  filename, _ = os.path.splitext(os.path.basename(img_path))

  subject_id, etc = filename.split('__')
  
  if real:
      gender, lr, finger, _ = etc.split('_')
  else:
      gender, lr, finger, _, _ = etc.split('_')
  
  gender = 0 if gender == 'M' else 1

  return np.array([subject_id], dtype=np.uint16)

def extract_gender(img_path,real):
  filename, _ = os.path.splitext(os.path.basename(img_path))

  subject_id, etc = filename.split('__')
  
  if real:
      gender, lr, finger, _ = etc.split('_')
  else:
      gender, lr, finger, _, _ = etc.split('_')
  
  gender = 0 if gender == 'M' else 1

  return np.array([gender], dtype=np.uint16)


def loading_data(path,real):
    print("loading data from: ",path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            img_resize = cv2.resize(img_array, (img_size, img_size))
            subject_id = extract_id(os.path.join(path, img), real)
            gender = extract_gender(os.path.join(path, img), real)
            data.append([subject_id[0], gender[0], img_resize])
        except Exception as e:
            pass
    data
    print("loaded: ", len(data))
    return data



Real_path = "SOCOFing/Real"
Easy_path = "SOCOFing/Altered/Altered-Easy"
Hard_path = "SOCOFing/Altered/Altered-Hard"

Easy_data = loading_data(Easy_path, train = True)
Hard_data = loading_data(Hard_path, train = True)
Real_data = loading_data(Real_path, train = False)

data = np.concatenate([Easy_data, Hard_data, Real_data], axis=0)

data = data[data[:,0].argsort()]
gender = data[:, 1]
numb_male = len(gender[gender == 0])
numb_female = len(gender[gender == 1])
print('Broj muskaraca: ', numb_male)
print('Broj zena: ', numb_female)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
lable= ['Muskarci', ' Zene']
numb = [numb_male, numb_female]
ax.bar(lable[0], numb[0])
ax.bar(lable[1], numb[1], color = 'red')
plt.show()

del Easy_data, Hard_data, Real_data, Real_path, Easy_path, Hard_path

split_test =int(0.2513 * len(data))
split_train = len(data) - split_test

train_data = data[:split_train, :]
test_data = data[split_train:, :]

random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(train_data)
random.shuffle(test_data)



img, train_labels = [], []
for subject_id, label, feature in train_data:
    train_labels.append(label)
    img.append(feature)
train_data = np.array(img).reshape(-1, img_size, img_size, 3)
train_data = train_data / 255.0

train_labels = to_categorical(train_labels, num_classes = 2)



del img
img, test_labels = [], []
for subject_id, label, feature in test_data:
    test_labels.append(label)
    img.append(feature)
test_data = np.array(img).reshape(-1, img_size, img_size, 3)
test_data = test_data / 255.0

test_labels = to_categorical(test_labels, num_classes = 2)


del data, img, subject_id, label, feature

base_model = VGG16(weights='imagenet', include_top=False, input_shape = [img_size, img_size, 3])

vgg = Sequential()
vgg.add(base_model)
vgg.add(Flatten())
vgg.add(Dense(2, activation = 'sigmoid'))

vgg.summary()

vgg.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = vgg.fit(train_data, train_labels, batch_size = 128, epochs = 10, 
          validation_split = 0.2)


pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)


res = vgg.evaluate(test_data, test_labels)
print("    Test Loss: {:.5f}".format(res[0]))
print("Test Accuracy: {:.2f}%".format(res[1] * 100))
pred = vgg.predict(test_data)
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
    plt.imshow(test_data[i], cmap='gray')

