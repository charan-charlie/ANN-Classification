import streamlit as st
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle



## Load the trained model

model = tf.keras.models.load_model("model.h5")

## Load the encoders and scaler

with open("label_encoder.pkl",'rb') as f:
    label_encoder_gender = pickle.load(f)

with open("One_Hot_Encoder.pkl",'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open("StandardScaler.pkl",'rb') as f:
    scaler = pickle.load(f)


## Streamlit app

st.title("Customer Churn Prediction")


## Taking the input from the user

geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credite Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_Products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_number = st.selectbox("Is Active Member",[0,1])


## prepare the input data

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_Products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_number],
    'EstimatedSalary' : [estimated_salary]
})


## Now lets add the geography after applying the one hot encoding 

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_data = pd.DataFrame(geo_encoded,columns = one_hot_encoder_geo.get_feature_names_out(["Geography"]))


## Combine one-hot encoded columns with input data

input_data = pd.concat([input_data.reset_index(drop = True),geo_data],axis = 1)


## apply the standard scaler in order to use it in model to predict

input_data_scaled = scaler.transform(input_data)

prediction = float(model.predict(input_data_scaled)[0][0])

st.write(f'Churn Probability : {prediction:.2f}')

if prediction > 0.5:
    st.write("The customer is likely to churn.")
else :
    st.write("The customer is not likely to churn.")







