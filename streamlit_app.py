import pandas as pd

dataset = pd.read_csv('cityu10c_train_dataset.csv')
dataset.head()

features = ['Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'LoanAmount', 'LoanDuration']
target = ['LoanApproved']

X = dataset[features]
y = dataset[target]

X.head()

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import pickle

# Assuming you have X and y defined as in your previous code snippets

# Define categorical and numerical features
categorical_features = ['EmploymentStatus', 'EducationLevel']
numerical_features = ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration']

# Create transformers for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')) # Using OneHotEncoder for categorical features
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Fit the pipeline
pipeline.fit(X, y.values.ravel()) # Use ravel() to avoid DataConversionWarning

# Save the pipeline to a pickle file
with open('decision_tree_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Pipeline trained and saved to decision_tree_pipeline.pkl")

import sklearn
print(sklearn.__version__)
