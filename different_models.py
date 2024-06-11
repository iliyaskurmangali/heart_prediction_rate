import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('data/Heart_Disease_Prediction.csv')

# Defining the features and the target
X = df.drop(columns='Heart Disease')
y_no_encode = df['Heart Disease']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_no_encode)

# Train-Test-Validation split
from sklearn.model_selection import train_test_split

# Split the data into training+validation and test sets
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# Split the training+validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.1, random_state=1)

# Define the numerical transformer (StandardScaler)
num_transformer = StandardScaler()

# Define the ColumnTransformer to preprocess numerical features
preprocessor = ColumnTransformer([
    ('num_transformer', num_transformer, ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
       'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
       'Slope of ST', 'Number of vessels fluro', 'Thallium'])
])

# Define the classification models to evaluate
classification_models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier()
}
model_names = []
precisions = []

# Train and evaluate each model using the validation set
for name, clf in classification_models.items():
    pipeline = make_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_val, y_val)  # Scoring based on precision
    model_names.append(name)
    precisions.append(score)
    print(f"{name} accuracy: {score:.2f}")

# Based on the evaluation, GaussianNB was found to be the best model

# Define the parameter grid for GaussianNB
param_grid = {
    'gaussiannb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Create the GridSearchCV object for hyperparameter tuning
grid_search = GridSearchCV(
    make_pipeline(preprocessor, GaussianNB()),
    param_grid=param_grid,
    cv=5,
    scoring="precision"
)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
grid_search.best_params_

print(grid_search.best_params_)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
from sklearn.metrics import classification_report

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))



import joblib
# Save the preprocessor and the best model
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(best_model, 'best_model.joblib')
