# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load your heart disease dataset
heart_data = pd.read_csv('heart.csv')

# Assume 'disease' is the column indicating the presence or absence of heart disease
X = heart_data.drop('disease', axis=1)
y = heart_data['disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters to search
LR_hp = {'C': np.logspace(-4, 4, 20),
         'solver': ['liblinear', 'lbfgs']}

# Hyperparameter tuning using RandomizedSearchCV
lr = RandomizedSearchCV(LogisticRegression(random_state=0),
                        param_distributions=LR_hp,
                        cv=5,
                        n_iter=20,
                        verbose=True)

# Fit the model
lr.fit(X_train_scaled, y_train)

# Streamlit web application
st.title('Heart Disease Prediction App')

# Sidebar with user input
st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', min_value=29, max_value=77, value=55)
    sex = st.sidebar.selectbox('Sex', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure', min_value=94, max_value=200, value=125)
    chol = st.sidebar.slider('Cholesterol', min_value=126, max_value=564, value=250)
    fbs = st.sidebar.checkbox('Fasting Blood Sugar > 120 mg/dl')
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=71, max_value=202, value=150)
    exang = st.sidebar.checkbox('Exercise Induced Angina')
    oldpeak = st.sidebar.slider('Oldpeak', min_value=0.0, max_value=6.2, value=2.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=1)
    thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])

    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal,
            }
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

# Display user input
st.subheader('User Input:')
st.write(user_input)

# Make predictions
prediction = lr.predict(scaler.transform(user_input))

# Display prediction
st.subheader('Prediction:')
if prediction[0] == 0:
    st.write("No heart disease.")
else:
    st.write("Heart disease present.")

if st.button('Re-run Model'):
    # Re-run the model with the same user input (optional)
    new_prediction = lr.predict(scaler.transform(user_input))
    st.subheader('New Prediction:')
    if new_prediction[0] == 0:
        st.write("No heart disease.")
    else:
        st.write("Heart disease present.")
