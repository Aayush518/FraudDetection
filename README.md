
# Credit Card Fraud Detection Streamlit App (Ongoing) 

This is a simple Streamlit web application for credit card fraud detection using a logistic regression model. The app allows users to input values for various features and predicts whether a credit card transaction is fraudulent or not.

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Screenshots](#screenshots)
- [Built With](#built-with)

## About the Project

The goal of this project is to demonstrate how to build a basic credit card fraud detection app using Streamlit. The app uses a pre-trained logistic regression model to make predictions based on user input.

## Getting Started

To run the app locally on your machine, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Aayush518/credit-card-fraud-detection-app.git
   ```

2. Navigate to the project directory:
   ```
   cd credit-card-fraud-detection-app
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit app in your web browser by following the instructions in the "Getting Started" section.

2. Use the sidebar to input values for the features related to the credit card transaction.

3. Click the "Predict" button to see the model's prediction for whether the transaction is fraudulent or not.

4. The app will display the prediction result and a confusion matrix heatmap.

## Features

- User-friendly interface for inputting credit card transaction features.
- Real-time prediction of whether a transaction is fraudulent or not.
- Visualization of confusion matrix heatmap to assess model performance.
- Histograms showing the distribution of selected features.


## Built With

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
