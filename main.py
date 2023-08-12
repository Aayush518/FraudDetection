import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')

std_scaler = StandardScaler()
data['scaled_amount'] = std_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = std_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

def main():
    st.title("Credit Card Fraud Detection")
    st.sidebar.header("User Input Features")

    user_input = {}

    user_input['V1'] = st.sidebar.number_input("V1", value=0.0)
    user_input['V2'] = st.sidebar.number_input("V2", value=0.0)


    if st.sidebar.button("Predict"):
        prediction = logreg.predict([list(user_input.values())])
        st.write("Prediction:", prediction)

    st.subheader("Confusion Matrix")
    cnf_matrix = confusion_matrix(y_test, y_pred.round())
    fig, ax = plt.subplots()
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

if __name__ == '__main__':
    main()
