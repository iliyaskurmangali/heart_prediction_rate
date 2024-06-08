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

df=pd.read_csv('Heart_Disease_Prediction.csv')

# Defining the features and the target

X = df.drop(columns='Heart Disease')
y_no_encode = df['Heart Disease']

label_encoder = LabelEncoder()
y=label_encoder.fit_transform(y_no_encode)

# Train-Test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35,random_state=2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Assuming num_transformer is defined, e.g., a StandardScaler
num_transformer = StandardScaler()
# Define the ColumnTransformer
preprocessor = ColumnTransformer([
    ('num_transformer', num_transformer, ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
       'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
       'Slope of ST', 'Number of vessels fluro', 'Thallium'])
])


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

for name, clf in classification_models.items():
    pipeline = make_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test, scoring='precison')
    model_names.append(name)
    precisions.append(score)
    print(f"{name} precision: {score:.2f}")


from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    make_pipeline(preprocessor, KNeighborsClassifier()),
    param_grid={'kneighborsclassifier__n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12]
    },
    cv=5,
    scoring="precision")

grid_search.fit(X_train, y_train)

grid_search.best_params_


best_model = grid_search.best_estimator_

from sklearn.metrics import classification_report, accuracy_score

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
