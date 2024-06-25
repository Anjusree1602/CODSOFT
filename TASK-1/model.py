import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Drop the Name and Ticket columns
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Fill missing Age values with the median Age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Cabin values with 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)

# Encode categorical variables using LabelEncoder
le_sex = LabelEncoder()
le_cabin = LabelEncoder()
le_embarked = LabelEncoder()
le_parch = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'].astype(str))
df['Cabin'] = le_cabin.fit_transform(df['Cabin'].astype(str))
df['Embarked'] = le_embarked.fit_transform(df['Embarked'].astype(str))
df['Parch'] = le_parch.fit_transform(df['Parch'].astype(int))

# Define the features and target variable
X = df.drop(['Survived', 'PassengerId'], axis=1)
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and the encoders to pickle files
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le_sex, open('le_sex.pkl', 'wb'))
pickle.dump(le_cabin, open('le_cabin.pkl', 'wb'))
pickle.dump(le_embarked, open('le_embarked.pkl', 'wb'))
pickle.dump(le_parch, open('le_parch.pkl', 'wb'))

# Save the feature names
pickle.dump(X.columns.tolist(), open('feature_names.pkl', 'wb'))

# Streamlit app
st.title("Titanic Survival Prediction")

# Create a sidebar with input fields
st.sidebar.header("User Input")
age = st.sidebar.slider("Age", 1, 100)
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
sibsp = st.sidebar.selectbox("SibSp", [0, 1])
pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
fare = st.sidebar.slider("Fare", 0, 100)
cabin = st.sidebar.selectbox("Cabin", ['Unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
embarked = st.sidebar.selectbox("Embarked", ['S', 'C', 'Q'])
parch = st.sidebar.slider("Parch", 0, 10)

# Create a button to predict the survival outcome
if st.button("Predict"):
    # Load the trained model and encoders
    model = pickle.load(open('model.pkl', 'rb'))
    le_sex = pickle.load(open('le_sex.pkl', 'rb'))
    le_cabin = pickle.load(open('le_cabin.pkl', 'rb'))
    le_embarked = pickle.load(open('le_embarked.pkl', 'rb'))
    le_parch = pickle.load(open('le_parch.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))

    # Create a pandas DataFrame from the user input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'SibSp': [sibsp],
        'Pclass': [pclass],
        'Fare': [fare],
        'Cabin': [cabin],
        'Embarked': [embarked],
        'Parch': [parch]
    })

    # Transform the categorical variables using the loaded LabelEncoders
    input_data['Sex'] = le_sex.transform(input_data['Sex'].astype(str))
    input_data['Cabin'] = le_cabin.transform(input_data['Cabin'].astype(str))
    input_data['Embarked'] = le_embarked.transform(input_data['Embarked'].astype(str))
    input_data['Parch'] = le_parch.transform(input_data['Parch'].astype(int))

    # Ensure the input data has the same feature names and order as the training data
    input_data = input_data[feature_names]

    # Use the model to predict the survival outcome
    prediction = model.predict(input_data)

    # Display the prediction result
    st.write("Survival Prediction:", "Survived" if prediction[0] == 1 else "Not Survived")
