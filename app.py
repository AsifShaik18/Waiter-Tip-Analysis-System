import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load(r"C:\Users\Shafi\Downloads\Waiter_Tips_prediction_model.pkl")

# Load existing data from the Excel file
try:
    existing_data = pd.read_excel("waiter_tips.xlsx")
except FileNotFoundError:
    existing_data = pd.DataFrame(columns=["Waiter Name", "Tip"])  # Create empty DataFrame if file doesn't exist

# Function to add new data
def add_new_data(waiter_name, tip):
    global existing_data  # Declare existing_data as a global variable
    
    # Check if existing_data is None or not initialized
    if existing_data is None or not isinstance(existing_data, pd.DataFrame):
        existing_data = pd.DataFrame(columns=["Waiter Name", "Tip"])  # Create an empty DataFrame
    
    # Check if waiter already exists in existing_data
    if waiter_name in existing_data["Waiter Name"].values:
        # Increment the tip amount for the existing waiter
        existing_data.loc[existing_data["Waiter Name"] == waiter_name, "Tip"] += tip
    else:
        # Append new data to existing_data
        new_data = pd.DataFrame({"Waiter Name": [waiter_name], "Tip": [tip]})
        existing_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    # Save DataFrame to Excel file
    existing_data.to_excel("waiter_tips.xlsx", index=False)

# Function to preprocess user input to match the training data encoding
def preprocess_user_input(df):
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'])
    expected_columns = ['total_bill', 'sex', 'smoker', 'day', 'time','size']
    
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[expected_columns]
    return df_encoded

# Streamlit UI for Waiter Tips Tracker
st.title("Waiter Tips Tracker")

# Input form to add new data
waiter_name = st.text_input("Enter Waiter Name:")
tip = st.number_input("Enter Tip Amount:", min_value=0.0, format="%.2f")

if st.button("Add Tip"):
    add_new_data(waiter_name, tip)
    st.success("Tip added successfully!")

# Display existing data
st.header("Existing Tips Data")
st.dataframe(existing_data)

if st.button("Clear All Data"):
    existing_data = pd.DataFrame(columns=["Waiter Name", "Tip"])  # Create an empty DataFrame
    existing_data.to_excel("waiter_tips.xlsx", index=False)
    st.success("All data cleared successfully!")
  
# Visualize the data
if not existing_data.empty:
    st.header("Tip Amounts by Waiter")

    # Bar graph
    fig, ax = plt.subplots()
    existing_data.groupby('Waiter Name')['Tip'].sum().plot(kind='bar', ax=ax)
    ax.set_ylabel("Total Tips")
    ax.set_title("Total Tips by Waiter")
    st.pyplot(fig)

    # Pie chart
    fig, ax = plt.subplots()
    existing_data.groupby('Waiter Name')['Tip'].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    ax.set_title("Tip Distribution by Waiter")
    st.pyplot(fig)

# Streamlit UI for Waiter Tips Prediction
st.title("Waiter Tips Prediction")

# Function to get user input
def get_user_input():
    total_bill = st.number_input("Total Bill", min_value=0.0, format="%.2f")
    sex = st.selectbox("Sex", options=["Female", "Male"])
    smoker = st.selectbox("Smoker", options=["No", "Yes"])
    day = st.selectbox("Day", options=["Thur", "Fri", "Sat", "Sun"])
    time = st.selectbox("Time", options=["Lunch", "Dinner"])
    size = st.number_input("Size", min_value=1, format="%d")

    user_data = {
        'total_bill': [total_bill],
        'sex': [sex],
        'smoker': [smoker],
        'day': [day],
        'time': [time],
        'size': [size]
    }
    return pd.DataFrame(user_data)

# Input Data
user_input_df = get_user_input()
st.header("User Input Data")
st.write(user_input_df)

# Prediction
if st.button("Predict"):
    preprocessed_input = preprocess_user_input(user_input_df)
    prediction = model.predict(preprocessed_input)
    st.write(f"Predicted Size: {prediction[0]}")