import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import numpy as np
import pandas as pd
import pickle

### Load the model
model = tf.keras.models.load_model("regression_model.h5")

### Load encoders and scalers
with open("label_encoder_gen.pkl","rb") as file:
    encoder_gen = pickle.load(file)

with open("one_hot_enc_geo.pkl","rb") as file:
    encoder_geo = pickle.load(file)

with open("regression_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

### Streamlit APP
st.title("Customer Estimated Salary Prediction")

## User Input
geography = st.selectbox("Geopgraphy",encoder_geo.categories_[0])
gender = st.selectbox("Gender",encoder_gen.classes_)
age = st.slider("Age",18,90)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score (CIBIL)")
exited = st.selectbox("Exited",[0,1])
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member?",[0,1])

### Prepare the input
input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[encoder_gen.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "Exited":[exited]
 })

### One hot encoding -Geography
encoded_geo = encoder_geo.transform([[geography]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo,columns = encoder_geo.get_feature_names_out(["Geography"]))

input_data = pd.concat([input_data.reset_index(drop=True),encoded_geo_df],axis=1)    ### reset_index(drop=True) will not create index column, which is anyway by default

### Scale the input
scaled_input = scaler.transform(input_data)

### Prediction
prediction =  model.predict(scaled_input)
predicted_salary = prediction[0][0]

st.write("Predicted Estimated Salary is ${}".format(predicted_salary))


