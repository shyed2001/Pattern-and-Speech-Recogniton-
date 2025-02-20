# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

print("All packages and modules are installed successfully!")
print("Hello, World!")
print("All packages and modules are installed successfully!")
# Load the datasets
train_data = pd.read_csv('fraudTrain100K.csv')
test_data = pd.read_csv('fraudTest10K.csv')

# Display basic information about the datasets
print("Train Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())

# Display the first few rows of the train dataset
print("\nTrain Data Sample:")
print(train_data.head())

# Check for missing values in both datasets
print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_data.isnull().sum())

# Removing rows with missing values
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Check the distribution of classes in the "is_fraud" column
class_distribution = test_data['is_fraud'].value_counts()
print(class_distribution)

# Make sure you have both classes (0 and 1) in your test data
if 0 in class_distribution and 1 in class_distribution:
    print("Both classes (0 and 1) are present in the 'is_fraud' column.")
else:
    print("One or both classes are missing from the 'is_fraud' column.")
    print("One or both classes are missing from the 'is_fraud' column.")
    print("One or both classes are missing from the 'is_fraud' column.")
    print("One or both classes are missing from the 'is_fraud' column.")
    print("One or both classes are missing from the 'is_fraud' column.")
    print("One or both classes are missing from the 'is_fraud' column.")
    print("One or both classes are missing from the 'is_fraud' column.")

# Display summary statistics of the train dataset
print("\nTrain Data Summary Statistics:")
print(train_data.describe())

# Visualize the distribution of the target variable (fraudulent or not)
plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=train_data)
plt.title('Distribution of Fraudulent Transactions , Please Close this window to continue using the program')
plt.xlabel('Is Fraud')
plt.ylabel('Count')
plt.show()

# Feature engineering and visualization 
# ... (code for feature engineering and visualization)
# Encoding categorical variables, feature scaling, and creating synthetic data
# 
# ... (code for encoding, scaling, and creating synthetic data)
# Split the resampled data into training and validation sets
# 
# ... (code for splitting the data into training and validation sets)
# Train and evaluate machine learning models
# 
# ... (code for training and evaluating various machine learning models)
# Comparison of results
# 
# ... (code for comparing and displaying the results)
# Feature engineering and visualization (similar to the code you provided)
# Additional feature engineering or visualization steps here
# Example: Extract hours and days from 'trans_date_trans_time'
train_data['trans_hour'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.hour
train_data['trans_day'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.dayofweek

# Example: Hourly distribution of fraud
plt.figure(figsize=(10, 6))
sns.countplot(x='trans_hour', hue='is_fraud', data=train_data)
plt.title('Hourly Distribution of Fraudulent Transactions, Please Close this window to continue using the program')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.legend(title='Is Fraud')
plt.show()

# Example: Day-wise distribution of fraud
plt.figure(figsize=(10, 6))
sns.countplot(x='trans_day', hue='is_fraud', data=train_data)
plt.title('Day-wise Distribution of Fraudulent Transactions, Please Close this window to continue using the program')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.legend(title='Is Fraud')
plt.show()

# Encoding categorical variables
encoder = OneHotEncoder(drop='first')
categorical_cols = ['gender', 'category', 'state']
encoded_train_features = encoder.fit_transform(train_data[categorical_cols]).toarray()
encoded_test_features = encoder.transform(test_data[categorical_cols]).toarray()

# Feature scaling
scaler = StandardScaler()
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
scaled_train_features = scaler.fit_transform(train_data[numerical_cols])
scaled_test_features = scaler.transform(test_data[numerical_cols])

# Concatenate encoded and scaled features for both train and test data
final_train_features = pd.concat([pd.DataFrame(encoded_train_features), pd.DataFrame(scaled_train_features)], axis=1)
final_test_features = pd.concat([pd.DataFrame(encoded_test_features), pd.DataFrame(scaled_test_features)], axis=1)

# Creating synthetic data (SMOTE)
smote = SMOTE(random_state=36)
x_train_resample, y_train_resample = smote.fit_resample(final_train_features, train_data['is_fraud'])

# Shuffle the resampled data
X_shuffled, y_shuffled = shuffle(x_train_resample, y_train_resample, random_state=42)

# Split the shuffled data into training and validation sets
x_train, x_validation, y_train, y_validation = train_test_split(X_shuffled, y_shuffled, test_size=0.5)

# For the initial selection process, we will use a smaller portion of the training dataset
x_train_copy = x_train
y_train_copy = y_train

x_train = x_train[:10000]
y_train = y_train[:10000]

# Example: Train and evaluate the Logistic Regression model
#lg_model = LogisticRegression()
lg_model = LogisticRegression(max_iter=500000)  # Increase the number of iterations as needed
lg_model.fit(x_train, y_train)
lg_predictions = lg_model.predict(x_validation)
lg_accuracy = accuracy_score(y_validation, lg_predictions)
print("Logistic Regression Accuracy: {:.3f}%".format(lg_accuracy * 100))

# Repeat the above code for other machine learning models such as SVM, KNN, RandomForest, MLP, etc.

# Calculate ROC curve and AUC
probs = lg_model.predict_proba(x_validation)[:, 1]
fpr, tpr, thresholds = roc_curve(y_validation, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) , Please Close this window to continue using the program')
plt.legend(loc="lower right")
plt.show()

from sklearn.svm import SVC

# Train SVM model with probability estimates enabled
svm_model = SVC(kernel='poly', probability=True)
svm_model.fit(x_train, y_train)

# Make predictions on test data
svm_predictions = svm_model.predict(x_validation)

# Calculate evaluation metrics on test data
svm_accuracy = accuracy_score(y_validation, svm_predictions)


# Print evaluation metrics with 3 decimal places, multiplied by 100
print("SVM Accuracy: {:.3f}%".format(svm_accuracy * 100))


# Train KNN model
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
# Make predictions on test data
knn_predictions = knn_model.predict(x_validation)

# Calculate evaluation metrics on test data
knn_accuracy = accuracy_score(y_validation, knn_predictions)


# Print evaluation metrics with 3 decimal places, multiplied by 100
print("KNN Accuracy: {:.3f}%".format(knn_accuracy * 100))

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
# Make predictions on test data
rf_predictions = rf_model.predict(x_validation)

# Calculate evaluation metrics on test data
rf_accuracy = accuracy_score(y_validation, rf_predictions)


# Print evaluation metrics with 3 decimal places, multiplied by 100
print("Random Forest Accuracy: {:.3f}%".format(rf_accuracy * 100))

# Train MLP model
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model.fit(x_train, y_train)

# Make predictions on test data
mlp_predictions = mlp_model.predict(x_validation)

# Calculate evaluation metrics on test data
mlp_accuracy = accuracy_score(y_validation, mlp_predictions)


# Print evaluation metrics with 3 decimal places, multiplied by 100
print("MLP Accuracy: {:.3f}%".format(mlp_accuracy * 100))

from sklearn.linear_model import SGDClassifier

# Train SGD model with 'log_loss' for probability estimates
sgd_model = SGDClassifier(loss='log_loss', random_state=42)
sgd_model.fit(x_train, y_train)


# Make predictions on test data
sgd_predictions = sgd_model.predict(x_validation)

# Calculate evaluation metrics on test data
sgd_accuracy = accuracy_score(y_validation, sgd_predictions)


# Print evaluation metrics with 3 decimal places, multiplied by 100
print("SGD Accuracy: {:.3f}%".format(sgd_accuracy * 100))

from sklearn.ensemble import ExtraTreesClassifier

# Train Extra Trees model
extra_trees_model = ExtraTreesClassifier(random_state=42)
extra_trees_model.fit(x_train, y_train)

# Make predictions on test data
ext_predictions = extra_trees_model.predict(x_validation)

# Calculate evaluation metrics on test data
ext_accuracy = accuracy_score(y_validation, ext_predictions)


# Print evaluation metrics with 3 decimal places, multiplied by 100
print("Extra Tree Accuracy: {:.3f}%".format(ext_accuracy * 100))


import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Define model names and instances
model_names = ['Logistic Regression', 'SVM', 'KNN', 'Random Forest', 'MLP', 'SGD', 'Extra Trees']
model_instances = [lg_model, svm_model, knn_model, rf_model, mlp_model, sgd_model, extra_trees_model]

# Initialize lists to store accuracy and ROC scores
accuracy_scores = []
roc_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

# Calculate accuracy and ROC scores for each model
for model in model_instances:
    predictions = model.predict(final_test_features)
    accuracy = accuracy_score(test_data['is_fraud'], predictions)  # Update 'test_target' to 'test_data['is_fraud']'
    roc_score = roc_auc_score(test_data['is_fraud'], predictions)  # Update 'test_target' to 'test_data['is_fraud']'
    accuracy_scores.append(accuracy)
    roc_scores.append(roc_score)
    f1_scores.append(f1_score(test_data['is_fraud'], predictions))
    precision_scores.append(precision_score(test_data['is_fraud'], predictions))
    recall_scores.append(recall_score(test_data['is_fraud'], predictions))

# Create a DataFrame to compare results
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy_scores,
    'ROC Score': roc_scores,
    'F1 Score': f1_scores,
    'Precision Score': precision_scores,
    'Recall Score': recall_scores,
})

# Print the comparison table
print(results_df)


# Plot ROC curves for all models on the same graph
plt.figure(figsize=(10, 6))
for model, model_name in zip(model_instances, model_names):
    probs = model.predict_proba(final_test_features)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_data['is_fraud'], probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='ROC curve of {}'.format(model_name))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC), Please Close this window to continue using the program')
plt.legend(loc="lower right")
plt.show()
#10 fold cross validation will take very very long time
# Calculate accuracy, ROC, and other scores for each model using 3-fold cross-validation
# Initialize a ThreadPoolExecutor
# Define the perform_cross_validation function here
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define model names and instances
model_names = ['Logistic Regression', 'SVM', 'KNN', 'Random Forest', 'MLP', 'SGD', 'Extra Trees']
model_instances = [lg_model, svm_model, knn_model, rf_model, mlp_model, sgd_model, extra_trees_model]

# Initialize lists to store accuracy and ROC scores
accuracy_scores = []
roc_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

# Create a StratifiedKFold object for 3-fold cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#10 fold cross validation will take very very long time
# Calculate accuracy, ROC, and other scores for each model using 10-fold cross-validation
for model, model_name in zip(model_instances, model_names):
    accuracy = cross_val_score(model, x_train_copy, y_train_copy, cv=cv, scoring='accuracy')
    roc = cross_val_score(model, x_train_copy, y_train_copy, cv=cv, scoring='roc_auc')
    f1 = cross_val_score(model, x_train_copy, y_train_copy, cv=cv, scoring='f1')
    precision = cross_val_score(model, x_train_copy, y_train_copy, cv=cv, scoring='precision')
    recall = cross_val_score(model, x_train_copy, y_train_copy, cv=cv, scoring='recall')
   # Create a DataFrame to compare results 
    accuracy_scores.append(accuracy.mean())
    roc_scores.append(roc.mean())
    f1_scores.append(f1.mean())
    precision_scores.append(precision.mean())
    recall_scores.append(recall.mean())
K = 3
print(f"{K}-fold cross validation results:")

# Print the comparison table
print(results_df)