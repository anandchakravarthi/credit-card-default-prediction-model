# UCI CREDIT CARD DEFAULT PREDICTION PROJECT
# Data Preprocessing, Exploratory Data Analysis and Machine Learning
# 0 = No Default, 1 = Default

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score, f1_score

df = pd.read_csv("C:\\Users\\Asus\\Downloads\\UPI\\UCI_Credit_Card.csv")

# Display the first five records
print("First five records:")
print(df.head(5))

# Display the last five records
print("\nLast five records:")
print(df.tail(5))

# Display dataset information, including columns, data types and non-null counts
print("\nDataset information:")
df.info()

# Display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Display the original number of rows and columns
print("\nOriginal dataset shape:", df.shape)

# REMOVING THE UNWANTED COLUMN

# Remove the customer ID because it is only an identifier
df.drop(columns=["ID"], inplace=True)

print("\nDataset shape after removing ID:", df.shape)

# DATA PREPROCESSING

# This dataset contains numerical and coded categorical values.
# No text standardization is required for these columns.

# DUPLICATE VALUE DETECTION AND REMOVAL

# Count and display duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows from the dataset
df.drop_duplicates(inplace=True)

print("Dataset shape after removing duplicates:", df.shape)

# MISSING VALUE ANALYSIS AND HANDLING

# Display the number of missing values before handling
print("\nMissing values before handling:")
print(df.isnull().sum())

# Select numerical columns automatically
numerical_columns = df.select_dtypes(include="number").columns

# Replace missing numerical values with the mean of each column
for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mean())

# Display the number of missing values after handling
print("\nMissing values after handling:")
print(df.isnull().sum())

# CORRELATION AND COVARIANCE ANALYSIS

# Calculate correlations between numerical columns
# Correlation measures the strength and direction of linear relationships
print("\nCorrelation matrix:")
print(df.corr(numeric_only=True).round(2))

# Calculate covariance between numerical columns
# Covariance indicates whether two variables tend to vary together
print("\nCovariance matrix:")
print(df.cov(numeric_only=True).round(2))

# SKEWNESS AND KURTOSIS ANALYSIS

# Calculate skewness for all numerical columns
print("\nSkewness of numerical columns:")
print(df.select_dtypes(include="number").skew().round(2))

# Calculate kurtosis for all numerical columns
print("\nKurtosis of numerical columns:")
print(df.select_dtypes(include="number").kurt().round(2))

# OUTLIER DETECTION USING THE IQR METHOD

# Select numerical columns automatically for outlier detection
outlier_columns = df.select_dtypes(include="number").columns

# Detect values outside the 1.5 x IQR boundaries in each numerical column
print("\nOutlier detection using IQR:")

for column in outlier_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    # Calculate the interquartile range
    IQR = Q3 - Q1
    # Identify outliers using the lower and upper boundaries
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    # Display the number of detected outliers
    print(f"Outliers in {column}: {len(outliers)}")

# EXPLORATORY DATA ANALYSIS AND VISUALIZATION

# VISUALIZATION 1: DISTRIBUTION OF CREDIT CARD DEFAULT

# Plot the number of customers who defaulted and did not default
plt.figure(figsize=(8, 5))
sns.countplot(data=df,
    x="default.payment.next.month",hue="default.payment.next.month",
    palette=["skyblue", "orange"],legend=False)
plt.title("Distribution of Credit Card Default")
plt.xlabel("Default Next Month (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# VISUALIZATION 2: AGE DISTRIBUTION

# Plot a histogram with a KDE curve to visualize customer age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["AGE"],bins=30,kde=True,color="cornflowerblue",edgecolor="black")
plt.title("Age Distribution of Credit Card Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# VISUALIZATION 3: DEFAULT BY GENDER

# Compare default and non-default customers by gender
plt.figure(figsize=(8, 5))
sns.countplot(data=df,x="SEX",hue="default.payment.next.month",palette=["skyblue", "orange"])
plt.title("Credit Card Default by Gender")
plt.xlabel("Gender (1 = Male, 2 = Female)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# VISUALIZATION 4: DEFAULT BY EDUCATION

# Compare default and non-default customers by education category
plt.figure(figsize=(9, 5))

sns.countplot(data=df,x="EDUCATION",hue="default.payment.next.month",palette=["skyblue", "orange"])
plt.title("Credit Card Default by Education Level")
plt.xlabel("Education Code")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# VISUALIZATION 5: DEFAULT BY MARRIAGE STATUS

# Compare default and non-default customers by marriage category
plt.figure(figsize=(8, 5))
sns.countplot(data=df,x="MARRIAGE",hue="default.payment.next.month",palette=["skyblue", "orange"])
plt.title("Credit Card Default by Marriage Status")
plt.xlabel("Marriage Code")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# VISUALIZATION 6: AVERAGE CREDIT LIMIT BY DEFAULT STATUS

# Calculate the average credit limit for each default category
limit_means = df.groupby("default.payment.next.month")["LIMIT_BAL"].mean()
plt.figure(figsize=(8, 5))
sns.barplot(x=limit_means.index.astype(str),y=limit_means.values,
    hue=limit_means.index.astype(str),palette=["skyblue", "orange"],legend=False)
plt.title("Average Credit Limit by Default Status")
plt.xlabel("Default Next Month (0 = No, 1 = Yes)")
plt.ylabel("Average Credit Limit")
plt.tight_layout()
plt.show()

# VISUALIZATION 7: CORRELATION HEATMAP

# Select all numerical features automatically for correlation analysis
selected_columns = df.select_dtypes(include="number").columns

# Calculate correlations between all numerical columns
corr = df[selected_columns].corr(numeric_only=True)

# Visualize correlations using a heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f",center=0,linewidths=0.5,annot_kws={"size": 7})
plt.title("Correlation Heatmap of Credit Card Features")
plt.tight_layout()
plt.show()

# VISUALIZATION 8: CREDIT LIMIT BY DEFAULT STATUS

# Use a boxplot to compare credit limits for default and non-default customers
plt.figure(figsize=(9, 5))
sns.boxplot(data=df,x="default.payment.next.month",y="LIMIT_BAL",hue="default.payment.next.month",
    palette=["#66B3FF", "#FF9999"],legend=False)
plt.title("Credit Limit by Default Status")
plt.xlabel("Default Next Month (0 = No, 1 = Yes)")
plt.ylabel("Credit Limit")
plt.tight_layout()
plt.show()

# FIVE OBJECTIVES OF THE CREDIT CARD DEFAULT PROJECT

# OBJECTIVE 1: ANALYZE OVERALL CREDIT CARD DEFAULT

# Count customers in each default category
print("\nOBJECTIVE 1: Overall Credit Card Default")
print(df["default.payment.next.month"].value_counts())

# Calculate the percentage of each default category
print("\nDefault percentage:")
print((df["default.payment.next.month"].value_counts(normalize=True) * 100).round(2))

# OBJECTIVE 2: STUDY DEFAULT BY GENDER

# Compare default counts for each gender
print("\nOBJECTIVE 2: Default by Gender")
print(pd.crosstab(df["SEX"], df["default.payment.next.month"]))

# Calculate the default percentage within each gender
print("\nPercentage within each gender:")
print((pd.crosstab(df["SEX"],df["default.payment.next.month"],normalize="index") * 100).round(2))

# OBJECTIVE 3: INVESTIGATE DEFAULT BY EDUCATION

# Compare default counts for each education category
print("\nOBJECTIVE 3: Default by Education")
print(pd.crosstab(df["EDUCATION"], df["default.payment.next.month"]))

# Calculate the default percentage within each education category
print("\nPercentage within each education group:")
print((pd.crosstab(df["EDUCATION"],df["default.payment.next.month"],normalize="index") * 100).round(2))

# OBJECTIVE 4: EXAMINE REPAYMENT STATUS AND DEFAULT

# Select repayment status columns
repayment_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Calculate average repayment status for each default category
print("\nOBJECTIVE 4: Average Repayment Status by Default")
print(df.groupby("default.payment.next.month")[repayment_columns].mean().round(2))

# OBJECTIVE 5: ANALYZE BILL AMOUNTS AND PAYMENT AMOUNTS

# Select monthly bill amount columns
bill_columns = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3","BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]

# Select monthly payment amount columns
payment_columns = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3","PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

# Compare average bill amounts and payment amounts by default category
print("\nOBJECTIVE 5: Average Bill Amounts and Payments by Default")
print(df.groupby("default.payment.next.month")[bill_columns + payment_columns].mean().round(2))


# FEATURE ENGINEERING

# Create total bill amount from six months of bill records
df["TOTAL_BILL_AMT"] = df[bill_columns].sum(axis=1)

# Create total payment amount from six months of payment records
df["TOTAL_PAY_AMT"] = df[payment_columns].sum(axis=1)

# Calculate average bill amount and average payment amount
df["AVG_BILL_AMT"] = df[bill_columns].mean(axis=1)
df["AVG_PAY_AMT"] = df[payment_columns].mean(axis=1)

# Calculate total and maximum repayment status
df["TOTAL_REPAYMENT_STATUS"] = df[repayment_columns].sum(axis=1)
df["MAX_REPAYMENT_DELAY"] = df[repayment_columns].max(axis=1)

# Calculate the ratio of total payments to total bills
df["PAYMENT_TO_BILL_RATIO"] = (df["TOTAL_PAY_AMT"] /df["TOTAL_BILL_AMT"].replace(0, np.nan))

# Replace undefined ratio values with zero
df["PAYMENT_TO_BILL_RATIO"] = (df["PAYMENT_TO_BILL_RATIO"].replace([np.inf, -np.inf], np.nan).fillna(0))

# Display the newly created features
print("\nNew features created:")
print(["TOTAL_BILL_AMT", "TOTAL_PAY_AMT", "AVG_BILL_AMT","AVG_PAY_AMT", "TOTAL_REPAYMENT_STATUS",
    "MAX_REPAYMENT_DELAY", "PAYMENT_TO_BILL_RATIO"])
print("\nDataset shape after feature engineering:", df.shape)

# PREPARE DATA FOR CLASSIFICATION

# Select input features and target variable
X = df.drop("default.payment.next.month", axis=1)
Y = df["default.payment.next.month"].astype(int)

# Convert coded categorical columns into dummy variables
X = pd.get_dummies(X,columns=["SEX", "EDUCATION", "MARRIAGE"],drop_first=True,dtype=int)

# Display target class distribution
print("\nTarget class distribution:")
print(Y.value_counts())

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
print("\nTraining dataset shape:", X_train.shape)
print("Testing dataset shape:", X_test.shape)

# Standardize the features for Logistic Regression and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CLASSIFICATION MODELS

# CLASSIFICATION MODEL 1: LOGISTIC REGRESSION

# Create and train the Logistic Regression model
model1 = LogisticRegression(max_iter=2000,class_weight="balanced")
model1.fit(X_train_scaled, Y_train)

# Predict credit card default categories
y_pred1 = model1.predict(X_test_scaled)

# Evaluate the Logistic Regression model
print("\nEvaluation parameters for Logistic Regression model")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred1, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred1, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred1))

print("\nClassification Report:")
print(classification_report(Y_test, y_pred1, zero_division=0))

# Plot the Logistic Regression confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(Y_test, y_pred1),annot=True,fmt="d",cmap="Blues",
    xticklabels=["No Default", "Default"],yticklabels=["No Default", "Default"])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# CLASSIFICATION MODEL 2: DECISION TREE

# Create and train the Decision Tree model
model2 = DecisionTreeClassifier(max_depth=10,random_state=42,class_weight="balanced")
model2.fit(X_train, Y_train)

# Predict credit card default categories
y_pred2 = model2.predict(X_test)

# Evaluate the Decision Tree model
print("\nEvaluation parameters for Decision Tree model")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred2, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred2, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred2))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred2, zero_division=0))

# Visualize the first three levels of the Decision Tree
plt.figure(figsize=(18, 9))

plot_tree(model2,feature_names=X.columns,class_names=["No Default", "Default"],
    filled=True,rounded=True,max_depth=3,fontsize=7)
plt.title("Decision Tree (First Three Levels)")
plt.tight_layout()
plt.show()

# CLASSIFICATION MODEL 3: RANDOM FOREST

# Create and train the Random Forest model
model3 = RandomForestClassifier(n_estimators=150,max_depth=15,random_state=42,n_jobs=-1,
    class_weight="balanced_subsample")
model3.fit(X_train, Y_train)

# Predict credit card default categories
y_pred3 = model3.predict(X_test)

# Evaluate the Random Forest model
print("\nEvaluation parameters for Random Forest model")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred3, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred3, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred3))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred3, zero_division=0))

# Display the most important features identified by Random Forest
feature_importance = pd.DataFrame({"Feature": X.columns,"Importance": model3.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print("\nTop 15 Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Plot the top 15 important features
plt.figure(figsize=(10, 7))
sns.barplot(data=feature_importance.head(15),x="Importance",y="Feature",color="steelblue")
plt.title("Top 15 Features - Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# CLASSIFICATION MODEL 4: SUPPORT VECTOR MACHINE

# Train SVM on a random sample to keep execution time practical
svm_sample_size = min(8000, len(X_train_scaled))
rng = np.random.RandomState(42)
svm_indices = rng.choice(len(X_train_scaled),size=svm_sample_size,replace=False)
X_svm_train = X_train_scaled[svm_indices]
Y_svm_train = Y_train.iloc[svm_indices]

# Create and train the Support Vector Machine model
model4 = SVC(kernel="linear",class_weight="balanced")
model4.fit(X_svm_train, Y_svm_train)

# Predict credit card default categories
y_pred4 = model4.predict(X_test_scaled)

# Evaluate the Support Vector Machine model
print("\nEvaluation parameters for Support Vector Machine model")
print("SVM training sample size:", svm_sample_size)
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred4, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred4, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred4))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred4, zero_division=0))

# MODEL ACCURACY COMPARISON

# Store the names of the four classification models
models = ["Logistic Regression","Decision Tree","Random Forest","Support Vector Machine"]

# Display evaluation parameters for Logistic Regression
print("\nMODEL 1: Logistic Regression")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred1, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred1, zero_division=0))

# Display evaluation parameters for Decision Tree
print("\nMODEL 2: Decision Tree")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred2, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred2, zero_division=0))

# Display evaluation parameters for Random Forest
print("\nMODEL 3: Random Forest")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred3, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred3, zero_division=0))

# Display evaluation parameters for Support Vector Machine
print("\nMODEL 4: Support Vector Machine")
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred4, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred4, zero_division=0))

# Create a model comparison table
predictions = [y_pred1, y_pred2, y_pred3, y_pred4]
model_comparison = pd.DataFrame({
    "Model": models,
    "Accuracy": [
        accuracy_score(Y_test, prediction)
        for prediction in predictions
    ],
    "Precision": [
        precision_score(Y_test, prediction, zero_division=0)
        for prediction in predictions
    ],
    "Recall": [
        recall_score(Y_test, prediction, zero_division=0)
        for prediction in predictions
    ],
    "F1 Score": [
        f1_score(Y_test, prediction, zero_division=0)
        for prediction in predictions
    ]})

# Sort models according to F1 score
model_comparison = model_comparison.sort_values(by="F1 Score",ascending=False)
print("\nMODEL COMPARISON:")
print(model_comparison.to_string(index=False))

# Identify the best-fit model using F1 score
best_model_name = model_comparison.iloc[0]["Model"]
best_accuracy = model_comparison.iloc[0]["Accuracy"]
best_f1 = model_comparison.iloc[0]["F1 Score"]
print("\nBEST-FIT MODEL:", best_model_name)
print("Best model accuracy:", round(best_accuracy, 4))
print("Best model F1 score:", round(best_f1, 4))

# KEY FINDINGS
print("\nKEY FINDINGS")
print("1. The dataset contains customer credit, repayment, bill and payment information.")
print("2. The target variable identifies whether a customer defaults on payment next month.")
print("3. Repayment status, credit limit, bill amounts and payment amounts can help explain default patterns.")
print("4. The Random Forest feature importance analysis identifies the features that contribute most to its predictions.")
print("5. The four classification models are compared using accuracy, precision,recall and F1 score.")

# FINAL CONCLUSION
print("\nFINAL CONCLUSION")
print(
    "Four machine learning models were developed to predict credit card default: "
    "Logistic Regression, Decision Tree, Random Forest and Support Vector Machine. "
    "The models were evaluated using accuracy, precision, recall and F1 score. "
    "The model with the highest F1 score was selected as the best-fit model.")

# RECOMMENDATIONS
print("\nRECOMMENDATIONS")
print("1. Monitor customers with repeated delayed repayment records.")
print("2. Review credit limits using customer repayment and payment behaviour.")
print("3. Use recall and F1 score along with accuracy when evaluating default detection.")
print("4. Investigate the important features identified by the Random Forest model.")
print("5. Validate the selected model using newer customer data before deployment.")

# LIMITATION
print("\nLIMITATION")
print(
    "The dataset contains historical customer information. Model performance "
    "on newer customers may differ, so further validation is required.")