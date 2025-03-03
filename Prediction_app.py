import streamlit as st
import pandas as pd 
import joblib
import numpy as np

pipeline = joblib.load('Pipeline.pkl')


st.title('Prediction App')
# st.write(pipeline.feature_names_in_)
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            color: #333333;
        }
        .main {
            background-color: white;
            background-image: url(https://github.com/Mikhthad/diabetes_prediction/blob/master/3701981.webp);
            background-size: cover;
            background-blend-mode: overlay;
            background-color: rgba(1, 1, 1, 0.002); /* Adjust transparency */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: black !important;
            text-align: center;
        }
        /* Style button */
        .stButton>button {
            background-color: black !important;  /* Default black background */
            color: white !important;            /* White text */
            font-size: 18px;
            border-radius: 5px;
            padding: 8px 20px;
            border: none;
            transition: background-color 0.3s ease;
        }
        /* Keep button black even on hover */
        .stButton>button:hover {
            background-color: black !important; /* Remains black */
            color: white !important;           /* Text stays white */
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)



# Main App Layout
st.markdown("""
    <style>
        .custom-text {
            font-size: 20px; /* Adjust text size */
            font-weight: bold; /* Make text bold */
            color: black !important; /* Force font color */
            font-family: 'Times New Roman', serif; /* Set font style */
            text-align: peragraph; /* Center align text */
        }
    </style>
    <p class="custom-text">This web app helps predict the likelihood of diabetes in females based on key health parameters. 
    Enter your details, and our model will analyze the data to provide an assessment. 
    Take a step towards better health with this simple and efficient tool!</p>
""", unsafe_allow_html=True)

# Apply custom CSS to set question background black and text white
st.markdown("""
    <style>
        /* Style question labels (background black, text white) */
        label {
            background-color: black !important; /* Black background */
            color: white !important; /* White text */
            font-size: 18px;
            font-weight: bold;
            font-family: 'Times New Roman', serif;
            padding:  2px 4px; /* Add some padding */
            border-radius: 5px; /* Optional: Rounded corners */
            display: inline; /* Ensures background applies correctly */
        }

        /* Set entire page background (optional) */
        body, .stApp {
            background-color: white !important; /* Set page background to white */
        }
    </style>
""", unsafe_allow_html=True)

cols, cols1 = st.columns(2)

with cols:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
with cols1:
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f")
    diabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)

user_data = np.array([[pregnancies,glucose,insulin,bmi,diabetesPedigreeFunction,age]])
predicts = pipeline.predict(user_data)
pd = pd.Series(predicts[0])
if st.button('predict'):
    # Display the predicted price with black font
    st.markdown(f"""
        <p style="font-size: 20px; font-weight: bold; color: black;">
            Predicted Result of the Given Data is {pd[0]}
        </p>
    """, unsafe_allow_html=True)


   
