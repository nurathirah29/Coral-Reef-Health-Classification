import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os

path = 'C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image/train'
classes = {'healthy': 0, 'bleached': 1}

import cv2

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

#plt.imshow(X[0], cmap='gray')
plt.imshow(X[10])

X_updated = X.reshape(len(X), -1)
X_updated.shape

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=0.20)
print(xtrain.shape, xtest.shape)

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

from sklearn.decomposition import PCA

print(xtrain.shape, xtest.shape)

pca = PCA(.98)

pca_train = pca.fit_transform(xtrain)
pca_test = pca.transform(xtest)
#pca_train = xtrain
#pca_test = xtest
print(pca_train.shape, pca_test.shape)
#print(pca.n_components_)
#print(pca.n_features_)

from sklearn.svm import SVC
import pickle

import warnings 
warnings.filterwarnings ('ignore')

model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
model.fit(xtrain, ytrain)

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training Score: {:.2f}%".format(model.score(xtrain, ytrain) * 100))
print("Testing Score: {:.2f}%".format(model.score(xtest, ytest) * 100))

pred = model.predict(xtest)

accuracy = accuracy_score(ytest, pred)
print("Testing Accuracy: {:.2f}%".format(accuracy * 100))

class_report = classification_report(ytest, pred, target_names=["Healthy", "Bleached"])
print("Classification Report:")
print(class_report)

misclassified = np.where(ytest!=pred)
misclassified
print("Total Misclassified Samples: ", len(misclassified[0]))

confusion_matrix = confusion_matrix(ytest, pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Healthy", "Bleached"])
cm_display.plot()
plt.show()

dec = {0: 'Healthy', 1:'Bleached'}

directory = 'C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image/test/healthy'

# Get the list of image files in the directory
image_files = os.listdir(directory)

# Visualize the images
plt.figure(figsize=(12, 10))
for i, image_file in enumerate(image_files[:12]):
    plt.subplot(4, 4, i+1)
    
    # Read the image
    img = cv2.imread(os.path.join(directory, image_file), 0)
    
    # Resize the image
    img_resized = cv2.resize(img, (200, 200))
    
    # Normalize the image
    img_normalized = img_resized/255
    
    # Reshape the image
    img_reshaped = img_normalized.reshape(1, -1)
    
    # Make predictions
    prediction = model.predict(img_reshaped)
    confidence = model.predict_proba(img_reshaped).max()  # Get the maximum probability as confidence
    
    # Map the predicted label to a class name
    predicted_class = dec[prediction[0]]
    
    # Set the title and display the image
    plt.title(f'Classification: {predicted_class}\nConfidence: {confidence:.2f}', color='black')
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Specify the directory path
directory = 'C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image/test/bleached'

# Get the list of image files in the directory
image_files = os.listdir(directory)

# Visualize the images
plt.figure(figsize=(12, 10))
for i, image_file in enumerate(image_files[:12]):
    plt.subplot(4, 4, i+1)
    
    # Read the image
    img = cv2.imread(os.path.join(directory, image_file), 0)
    
    # Resize the image
    img_resized = cv2.resize(img, (200, 200))
    
    # Normalize the image
    img_normalized = img_resized / 255
    
    # Reshape the image
    img_reshaped = img_normalized.reshape(1, -1)
    
    # Make predictions
    prediction = model.predict(img_reshaped)
    confidence = model.predict_proba(img_reshaped).max()  # Get the maximum probability as confidence
    
    # Map the predicted label to a class name
    predicted_class = dec[prediction[0]]
    
    # Set the title and display the image
    plt.title(f'Classification: {predicted_class}\nConfidence: {confidence:.2f}', color='black')
    plt.imshow(img)
    plt.axis('off')
plt.show()