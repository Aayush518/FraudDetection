import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Preprocess the data
def preprocess_data(data):
    std_scaler = StandardScaler()
    data['scaled_amount'] = std_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['scaled_time'] = std_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    return data

def train_model(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    return logreg, X_test, y_test, y_pred

def main():
    st.title("Credit Card Fraud Detection")
    st.sidebar.header("User Input Features")

    # Collect user input for features
    user_input = {}
    for col in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']:
        user_input[col] = st.sidebar.number_input(col, value=0.0)

    # Load and preprocess the data
    processed_data = preprocess_data(data.copy())

    # Add new features to the processed data
    for col, value in user_input.items():
        processed_data[col] = value

    # Train the logistic regression model
    logreg, X_test, y_test, y_pred = train_model(processed_data)

    # Perform prediction on user input
    if st.sidebar.button("Predict"):
        prediction = logreg.predict([list(user_input.values())])
        st.write("Prediction:", prediction[0])  # Assuming prediction is an array, take the first element

        # Update predictions with new features and plot confusion matrix heatmap
        y_pred_new = logreg.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, y_pred_new.round())
        fig, ax = plt.subplots()
        sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    # Display distribution of features
    st.sidebar.subheader("Feature Distributions")
    for col in user_input.keys():
        st.sidebar.write(f"**{col}**")
        plt.figure(figsize=(6, 4))
        sns.histplot(processed_data[col], bins=30, kde=True)
        plt.xlabel(col)
        plt.ylabel("Density")
        st.sidebar.pyplot(plt)

if __name__ == '__main__':
    main()
