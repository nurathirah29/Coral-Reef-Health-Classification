import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2

path = 'C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image 2/train'
classes = {'healthy': 0, 'bleached': 1}

X = []
Y = []

for cls, label in classes.items():
    class_path = os.path.join(path, cls)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path, 0)
        img_resized = cv2.resize(img, (200, 200))
        X.append(img_resized)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

# Visualize data
plt.imshow(X[0], cmap='gray')

# Prepare data
X_updated = X.reshape(len(X), -1)
X_updated.shape

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=0.20)
print(xtrain.shape, xtest.shape)

#feature scaling
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

#feature selection: PCA
from sklearn.decomposition import PCA

print(xtrain.shape, xtest.shape)
pca = PCA(.98)
#pca_train = pca.fit_transform(xtrain)
#pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest
#print(pca_train.shape, pca_test.shape)
#print(pca.n_components_)
#print(pca.n_features_)

#Train model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings 
warnings.filterwarnings ('ignore')

lg = LogisticRegression (C=0.1)
lg.fit(pca_train, ytrain)

sv = SVC()
sv.fit(pca_train, ytrain)

#Evaluation
print("Training Score:", lg.score(pca_train,ytrain))
print("Testing Score:", lg.score(pca_test,ytest))

print("Training Score:", sv.score(pca_train,ytrain))
print("Testing Score:", sv.score(pca_test,ytest))

#Prediction
pred = sv.predict(pca_test)
np.where(ytest!=pred)

pred[6]
ytest[6]

#Test model
dec = {0: 'Healthy', 1:'Bleached'}

plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image 2/test')
c=1
for i in os.listdir('/C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image 2/test/healthy/')[:9]:
    plt.subplot(3,3,c)

    img = cv2.imread('/C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image 2/test/healthy/' + i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir('/C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image 2/test/')
c=1
for i in os.listdir('/C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image 2/test/bleached/')[:16]:
    plt.subplot(4,4,c)

    img = cv2.imread('/C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image 2/test/bleached/' + i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    c+=1
