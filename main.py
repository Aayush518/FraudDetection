import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

# Load the dataset from the csv file
data = pd.read_csv('creditcard.csv')

# Understand the structure of the data
print(data.columns)
print(data.shape)
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Look at the class distribution for fraud vs. non-fraud transactions
print(data['Class'].value_counts())

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
std_scaler = StandardScaler()

data['scaled_amount'] = std_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = std_scaler.fit_transform(data['Time'].values.reshape(-1,1))

# Now drop the original columns (Amount and Time)
data = data.drop(['Time','Amount'], axis=1)

# Define the feature set and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Instantiate the model (using the default parameters)
logreg = LogisticRegression()

# Fit the model with data
logreg.fit(X_train, y_train)

# Predict on the test data
y_pred=logreg.predict(X_test)

# Check the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Check the classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')