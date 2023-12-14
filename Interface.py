import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as im

# Load your pre-trained model
model = load_model('./classifier_model.h5')

st.title("Deep Lung Classifier (Normal/Covid/Pneumonia)")

uploaded_file = st.file_uploader("Choose an image",type="png")


def predict_img(img_path):
    # Load the imag
    img = im.load_img(img_path, target_size=(256, 256))

    #Convert the image to a NumPy array
    img_array = im.img_to_array(img)

    # Expand the dimensions of the image
    img_array = np.expand_dims(img_array, axis=0)

    # Rescale the pixel values to be in the range [0, 1]
    img_array /= 255.0

    #Predict the image class
    predictions = model.predict(img_array)

    #  Class labels
    labels = ['COVID', 'Normal', 'Pneumonia']

    max_index = np.argmax(predictions)

    confidence = round(predictions[0,max_index] * 100,2)

    # Display the predicted class
    predicted_class = labels[max_index]

    return predicted_class, confidence



if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Save the uploaded image locally
    save_button = st.button("Predict Image")
    if save_button:
        image = Image.open(uploaded_file)
        img_path = "./UploadedImages/"+uploaded_file.name
        image.save(img_path)
        predicted_class, confidence = predict_img(img_path)
        st.success("Classification : " + predicted_class + ", with a confidence of : " + confidence.astype(str) + "%")
