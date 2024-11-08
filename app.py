import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Load the model saved in HDF5 format
classifier = load_model('D:\Data_Science_Projects\PlantVillage2\mobilenet_model-v1.h5')

def welcome():
    return "Welcome to the Animal Classification App!"

import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# Load the pre-trained MobileNet model (assuming it's saved in HDF5 format)
model = load_model('D:\Data_Science_Projects\PlantVillage2\mobilenet_model-v1.h5')  # Load your saved model globally

def predict_plant_disease(image):
    """
    Predict the plant disease from an image using the pre-loaded MobileNet model.
    
    Args:
        image (PIL.Image): The image of the plant to be predicted.
    
    Returns:
        str: Predicted disease label.
    """
    
    # Preprocess the image
    image = image.resize((224, 224))  # Resize to 224x224, the input size expected by MobileNet
    image = keras_image.img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, height, width, channels)
    image = image / 255.0  # Normalize pixel values to [0, 1]

    # Predict the class probabilities
    prediction = model.predict(image)
    
    # Map prediction to class labels
    class_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
                    'Potato___healthy', 'Potato___Late_blight', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
                    'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
                    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
                    'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

    predicted_class = class_labels[np.argmax(prediction)]  # Get the predicted class label
    return predicted_class



def main():
    st.title('Plant Disease Predictor')
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Plant Disease Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # File uploader for plant image
    uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Predict the disease if the user clicks the Predict button
        if st.button("Predict"):
            result = predict_plant_disease(image)
            st.success(f'The predicted disease is: {result}')
    
    # About section
    if st.button("About"):
        st.text("Plant Disease Prediction App")
        st.text("Built with Streamlit and MobileNet")

if __name__ == '__main__':
    main()
    