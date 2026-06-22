import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load data and train the SVM model
def train_svm_model():
    # Prepare/collect data
    path = 'C:/Users/Aqilah/Downloads/DEGREE/FYP/PROJECT/Coral Image/train'
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

    # Prepare data
    X_updated = X.reshape(len(X), -1)

    # Feature scaling
    xtrain = X_updated / 255

    # Feature selection: PCA
    pca = PCA(.98)
    pca_train = pca.fit_transform(xtrain)

    # Train model
    sv = SVC(probability=True)  # Set probability parameter to True
    sv.fit(pca_train, Y)

    return sv, pca

# Streamlit app
def main():
    # Set custom background image style
    custom_style = """
        <style>
        [data-testid="stAppViewContainer"] {
            
            background-size: cover;
            background-repeat: no-repeat;
        }
        .header-layer {
            background-color: rgba(255, 255, 255, 0.6);
            padding: 10px;
            border-radius: 10px;
        }
        </style>
    """
    st.markdown(custom_style, unsafe_allow_html=True)

    # Set header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('')
    with col2:
        st.markdown("<div class='header-layer'><h1 style='text-align: center; color: coral;'>Coral Reef</h1></div>", unsafe_allow_html=True)
    with col3:
        st.write('')

    # Load SVM model
    sv, pca = train_svm_model()

    # Image uploader
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the uploaded image
        img = np.array(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (200, 200))
        img_normalized = img_resized / 255
        img_reshaped = img_normalized.reshape(1, -1)
        img_pca = pca.transform(img_reshaped)

        # Make prediction
        prediction = sv.predict(img_pca)
        predicted_class = "Healthy" if prediction[0] == 0 else "Bleached"
        #prediction_prob = sv.predict_proba(img_pca)
        #confidence = np.max(prediction_prob) * 100

        # Display prediction result
        st.subheader("Prediction")
        st.write("Class: ", predicted_class)
        #st.write("Confidence: {:.2f}%".format(confidence))

if __name__ == '__main__':
    main()
