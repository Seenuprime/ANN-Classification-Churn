import pandas as pd
import tensorflow as tf
import streamlit as st 
import pickle


model = tf.keras.models.load_model(r"res/annchurn.keras")
# print(model.summary())
scaler = pickle.load(open('res/scaler.pkl', 'rb'))
geo_encoder = pickle.load(open('res/geo_encoder.pkl', 'rb'))
gender_encoder = pickle.load(open('res/gender_encoder.pkl', 'rb'))

CreditScore = st.text_input("Enter Credit Score: ")
gender = st.text_input("Enter Gender: ")
age = st.text_input("Enter your Age: ")
tenuer = st.text_input("Enter your Tenuer: ")
balance = st.text_input("Enter your bank Balance: ")
no_of_products = st.text_input("Enter number of Products: ")

options = ['yes', 'no']
hascrcard = st.radio(label="Do you have Credit Card: ", options=options)
# hascrcard = st.text_input("Do you have Credit Card: ")
active_member = st.radio(label="Active Member: ", options=options)
# active_member = st.text_input("Are you a active Member: ")

salary = st.text_input("Your Salary: ")
country = st.text_input("Country ")

df = pd.DataFrame({
    "CreditScore": [CreditScore],
    "Geography": [country],
    "Gender": [gender],
    "Age": [age],
    'Tenure': [tenuer],
    "Balance": [balance],
    "NumOfProducts": [no_of_products],
    "HasCrCard": [hascrcard],
    "IsActiveMember": [active_member],
    "EstimatedSalary": [salary]
})

st.write(df)
# df = df.apply(lambda x: x.lower()if isinstance(x,str) else x)

df['Gender'] = gender_encoder.transform(df['Gender'])
st.write(df)
names = geo_encoder.get_feature_names_out(['Geography'])
print(names)
store = geo_encoder.transform([df['Geography']])

df[names] = store.toarray()

df.drop('Geography', axis=1, inplace=True)

mapping = {'yes': 1, 'no': 0}
df['HasCrCard'] = df['HasCrCard'].map(mapping)
df['IsActiveMember'] = df['IsActiveMember'].map(mapping)

data = scaler.transform(df)

st.write(data)

preds = model.predict(data)
print(preds)

if preds[[0]] < 0.5:
    st.write('The Customer is not likely to leave!')
else:
    st.write('The Customer is likely to leave!')
