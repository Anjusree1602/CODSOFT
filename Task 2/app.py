import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title("Data Processing")

# Create a file uploader
uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv('advertising.csv')

    # Display the original data
    st.header("Original Data")
    st.write(df)

    # Check for missing values
    missing_values = df.isnull().sum()
    st.header("Missing Values")
    st.write(missing_values)

    # Drop any unnecessary columns
    df.dropna(axis=1, inplace=True)

    # Convert categorical variables to numerical variables
    categorical_cols = ['Month', 'Quarter']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    scaled_cols = ['TV', 'Radio', 'Newspaper']
    for col in scaled_cols:
        if col in df.columns:
            df[[col]] = scaler.fit_transform(df[[col]])

    # Display the processed data
    st.header("Processed Data")
    st.write(df)
