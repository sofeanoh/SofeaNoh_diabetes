#%%#1. IMPORT
import pickle
import os
import streamlit as st 
import numpy as np
import pandas as pd
from PIL import Image

#%% 2. CONSTANT
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model.pkl') #or can also do "models"
#IF WANT TO ADD IMAGE, SO HERE INSERT THE IMAGE PATH (static folder is where our resources images go thru)

IMAGE_PATH = os.path.join(os.getcwd(), 'static', 'header_image.jpg')
#%% 3. MODEL LOADING

with open(MODEL_PATH, 'rb') as file:
    classifier = pickle.load(file)
    
#%% before we want to deploy our model, we need to test our model again. (this is to be deleted after deployment)

new_data = [142, 168, 67, 25.3, 80]
print(classifier.predict(np.expand_dims(new_data, axis=0)))

#there is a controversial for DiabetesPedigreeFunction, unless there is a measurement for the DiabetesPedigreeFunction, 
#because everything else is able to be keyin by the user. but what exactly is pedgree? put urself in our user's shoes
# %% Streamlit Application Code

image = Image.open(IMAGE_PATH)
st.image(image, use_container_width=True)
st.title('Do you Have Diabetes?')
st.header("This is an application to help you predict whether you have diabetes or not.")
st.subheader('What do you know about diabetes?')
st.write('According to World Health Organisation (WHO), Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose.')

#adding sidebar
st.sidebar.header("Please input the information correctly here: ")
with st.sidebar:
    with st.form(key ='my_form'): #this will create a form
        Glucose = st.sidebar.slider('Glucose', 44, 199, 117) #means minimum is 44, maximum is 199, by default it is at 117
        BloodPressure = st.sidebar.slider('BloodPressure (mm/Hg)', 24, 250, 120)
        # SkinThickness = st.sidebar.number_input('What is your skin thickness?', 0, 100, 20)
        Insulin = st.sidebar.number_input('What is your insulin reading?', 0, 1000, 200)
        BMI = st.sidebar.number_input('What is your bmi reading', 18, 200, 25)
        # DiabetesPedigreeFunction = st.sidebar.slider('Your diabetes Pedegree?', 0.1, 10.0, 0.1)
        Age = st.sidebar.number_input('Your age', 0, 150, 18)
        submitted = st.form_submit_button("Submit")
        
st.subheader("Predicted Result")
if submitted:
    new_data = np.expand_dims([Glucose, BloodPressure, Insulin, BMI, Age], axis=0)
    outcome = classifier.predict(new_data)[0]
    if outcome == 0:
        st.success('CONGRATULATIONS!! YOU ARE FREE FROM DIABETES!! 🥳🥳🥳')
        st.write('Keep it up! ✨✨✨')
        st.balloons()
    
    else:
        st.warning('OH NO! YOU NEED TO WATCH OUT FOR YOUR DIET AND SPEAK TO OUR DOCTOR ☠️☠️☠️')
        st.write('🤡 Please reduce your sugar intake! Like Now!')
        st.snow()
# %% 
