import streamlit as st
import pandas as pd
import joblib
import pickle
# Load the trained Linear Regression model
model = joblib.load("crop_prediction_model.pkl")

# Streamlit app
st.title("Crop Predictor")

# Add an image to the page
#st.image("C:/Users/dinalik/Desktop/python course/istockphoto-1127372646-612x612.jpg")

# Sidebar with input fields for crop properties
st.sidebar.header("Enter crop Properties")
Nitrogen = st.sidebar.slider("Nitrogen", 0.0, 160.0, 8.0)
Pottasium = st.sidebar.slider("Pottasium", 5.0, 205.0, 0.5)
Phosphorus = st.sidebar.slider("Phosphorus", 5.0, 145.0, 0.5)
Temperature = st.sidebar.slider("Temperature", 8.0, 50.0, 10.0)
Humidity = st.sidebar.slider("Humidity", 14.0, 100.0, 0.08)
pH = st.sidebar.slider("pH", 3.0, 10.0, 30.0)
Rainfall = st.sidebar.slider("Rainfall", 20.0, 300.0, 100.0)

df= pd.read_csv('C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/Crop_recommendation.csv')
c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
crop_labels=list(targets.values())
crop_labels = [label.capitalize() for label in crop_labels]
# Predict Crop
crop_properties = [[Nitrogen, Pottasium, Phosphorus, Temperature,
                    Humidity, pH, Rainfall]]
predicted_label = model.predict(crop_properties)[0]
predicted_crop_name = crop_labels[predicted_label]  # Assuming the label is an index

image_paths = {
 0: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/apple.jfif',
 1: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/banana.jfif',
 2: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/blackgram.jfif',
 3: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/chickpea.jpg',
 4: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/coconut.jpg',
 5: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/coffee.jpg',
 6: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/cotton.jfif',
 7: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/grapes.jpg',
 8: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/jute.jpg',
 9: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/KIDNEYBEANS.jpg',
 10: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/lentils.jpg',
 11: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/maize.jfif',
 12: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/mango.jfif',
 13: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/mothbeans.jfif',
 14: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/mung.jpg',
 15: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/muskmelon.jfif',
 16: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/orange_1.jPG',
 17: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/papaya.jpg',
 18: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/pigeon.jpg',
 19: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/pomegranate.jfif',
 20: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/rice.jfif',
 21: 'C:/Users/dinalik/Desktop/python course/CROP_PREDICTOR/crops/watermelon.jpg'
 }

image_path = image_paths.get(predicted_label, None)
# Display predicted category
st.image(image_path, caption=f"Crop: {predicted_label}")
st.title (f"Predicted crop label: ")
st.title(predicted_crop_name)
