import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

df=pd.read_csv('Heart_Disease_Prediction.csv')

# Defining the features and the target

X = df.drop(columns='Heart Disease')
y_no_encode = df['Heart Disease']

label_encoder = LabelEncoder()
y=label_encoder.fit_transform(y_no_encode)

# Train-Test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Assuming num_transformer is defined, e.g., a StandardScaler
num_transformer = StandardScaler()
# Define the ColumnTransformer
preprocessor = ColumnTransformer([
    ('num_transformer', num_transformer, ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
       'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
       'Slope of ST', 'Number of vessels fluro', 'Thallium'])
])

pipeline = make_pipeline(preprocessor, LogisticRegression())

# Train Pipeline
pipeline.fit(X_train,y_train)

# Make predictions
pipeline.predict(X_test.iloc[0:1])

# Score model
pipeline.score(X_test,y_test)
# Cross-validate Pipeline
score_val=cross_val_score(pipeline, X_train, y_train, cv=5, scoring='precision').mean()
print(score_val)

train_sizes = [12, 24, 36,48,60,72, 84,96,108, 120, 132,144,180, 200]

# Get train scores (R2), train sizes, and validation scores using `learning_curve`
train_sizes, train_scores, test_scores = learning_curve(
estimator=pipeline, X=X, y=y, train_sizes=train_sizes, cv=5,scoring='precision')

# Take the mean of cross-validated train scores and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training score')
plt.plot(train_sizes, test_scores_mean, label = 'Test score')
plt.ylabel('precision', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves', fontsize = 18, y = 1.03)
plt.legend()
plt.show()
