import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

data = load_data()

# Handle missing values (if any)
data.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = StandardScaler()
data[['Amount', 'Time']] = scaler.fit_transform(data[['Amount', 'Time']])

# Handle class imbalance using SMOTE
X = data.drop('Class', axis=1)
y = data['Class']
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Train random forest classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Evaluate logistic regression model
y_pred_log_reg = log_reg.predict(X_test)
log_reg_report = classification_report(y_test, y_pred_log_reg)

# Evaluate random forest classifier
y_pred_rf_clf = rf_clf.predict(X_test)
rf_clf_report = classification_report(y_test, y_pred_rf_clf)

# Streamlit web app
st.title("Credit Card Fraud Detection")

# Display data sample
st.subheader("Data Sample")
st.write(data.sample(5))

# Display model evaluation
st.subheader("Logistic Regression Model Evaluation")
st.text(log_reg_report)

st.subheader("Random Forest Classifier Model Evaluation")
st.text(rf_clf_report)

# Allow users to input new transaction data
st.subheader("Predict New Transaction")
amount = st.number_input('Transaction Amount')
time = st.number_input('Transaction Time')
# Add inputs for other features as needed

if st.button('Predict'):
    input_data = [[amount, time]]  # Extend with other features
    input_data = scaler.transform(input_data)
    prediction = log_reg.predict(input_data)
    st.write('Fraudulent' if prediction[0] else 'Genuine')
