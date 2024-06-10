Introduction
The CardioHealth Risk Assessment Dataset provides a rich repository of medical and demographic data, facilitating the development and validation of machine learning models for heart disease prediction. This project endeavors to harness a diverse array of classification algorithms to precisely forecast the presence of heart disease based on patients' attributes and health indicators.

Dataset Overview
The dataset encompasses a multitude of features, including age, sex, chest pain type, blood pressure, cholesterol levels, and lifestyle factors, among others. These attributes serve as pivotal inputs for the predictive models, while the target variable, "Heart Disease," delineates the presence or absence of the condition. The dataset can be accessed on Kaggle here.

Installation
To replicate this project, ensure you have Python installed. Then, install the requisite dependencies using pip:

bash
Copy code
pip install -r requirements.txt
Data Preprocessing
The data preprocessing phase encompasses several steps:

Loading the dataset using pandas.
Separating the target variable and encoding it via LabelEncoder.
Splitting the dataset into training, validation, and test sets.
Standardizing numerical features using StandardScaler.
Model Evaluation
A diverse ensemble of classification models is trained and evaluated, including Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Gaussian Naive Bayes, and XGBoost. Evaluation metrics, particularly precision scores, are employed to gauge model performance on the validation set.

Model Performance
The precision scores on the validation set are delineated as follows:

Logistic Regression: 0.76
K-Nearest Neighbors: 0.84
Support Vector Machine: 0.80
Decision Tree: 0.60
Random Forest: 0.76
Gradient Boosting: 0.80
AdaBoost: 0.72
Gaussian Naive Bayes: 0.88
XGBoost: 0.76
Gaussian Naive Bayes emerges as the top performer based on the evaluation, exhibiting the highest precision.

Hyperparameter Tuning
Hyperparameter tuning is executed on the Gaussian Naive Bayes model employing GridSearchCV. This endeavor optimizes the model's hyperparameters, thereby enhancing its performance.

Final Evaluation
The tuned Gaussian Naive Bayes model is subjected to evaluation on the test set, resulting in an impressive accuracy of 93%. Furthermore, a comprehensive classification report is generated, elucidating precision, recall, and F1-score for each class
