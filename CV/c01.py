import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from skimage.feature import hog
from flask import Flask

img_list = []
label_list = []
with open('./flowerdata/17flowers/files.txt', 'r') as f:
    for img in f:
        img_list.append(img.rstrip())

for i in range(len(img_list)):
    label_list.append(i // 80)


# datas = pd.DataFrame({'imgs': img_list, 'labels': label_list})
feature_list = []
for i in img_list:
    path = os.path.join('./flowerdata/17flowers/', i)
    feature = cv.imread(path, 0)
    feature = cv.resize(feature, (50, 50))
    feature_list.append(hog(feature))

features = np.array(feature_list)
labels = np.array(label_list)

scaler = StandardScaler()
features = scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=12)
# model = RandomForestClassifier(n_estimators=200, max_features=6, random_state=12)
model = SGDClassifier(alpha=0.1)
model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print('train: ', train_score)
print('test: ', test_score)