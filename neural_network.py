import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras import datasets, layers, models
import os
import random
import pickle
CATEGORIES = [name.lower() for name in os.listdir('training_images')]

IMG_SIZE = 100

training_set = []

def create_training_set():
    for category in CATEGORIES:
        path = os.path.join('training_images', category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_set.append([new_array, class_num])
            except Exception as e:
                print(e)

create_training_set()
random.shuffle(training_set)
for sample in training_set[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_set:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 na koncu oznacza 1 kolor (3 bedzie dla rgb)

pickle_out = open('X_gray.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()


pickle_out = open('y_gray.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
