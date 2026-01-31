import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
pd.set_option('display.max_columns', None)
data = pd.read_csv('C://Users//uttam//OneDrive//Desktop//Analytics Projects//HR-Employee-payroll.csv')
data.head()
print(f'{data.shape[1]} features in total, each contains {data.shape[0]} data points')
data.describe()
data.info()
# Let's replace 'Attritition' , 'overtime' , 'Over18' column with integers before performing any visualizations
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Over18'] = data['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
data['OverTime'] = data['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
# Let's see if we have any missing data, luckily we don't!
if data.isnull().sum().sum() == 0:
    print('CHECK: No missing data \n')
else:
    print('CHECK: Missing data found \n')
print(data.isnull().sum())
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Attrition', data=data)
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.xticks([0,1], ['Stayed', 'Left'])
plt.annotate(f'{data["Attrition"].value_counts()[0]} \n {round(data["Attrition"].value_counts()[0]/len(data)*100,1)} %', (0, data["Attrition"].value_counts()[0]), ha='center', va='bottom')
plt.annotate(f'{data["Attrition"].value_counts()[1]}\n {round(data["Attrition"].value_counts()[1]/len(data)*100,1)}%', (1, data["Attrition"].value_counts()[1]), ha='center', va='bottom')

plt.show()
plt.show()
#plot histogram for each numeric variable/feature of the dataset
data.hist(figsize=(20,20), bins=50)
plt.show()
data.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis="columns", inplace=True)
left_df = data[data['Attrition'] == 1]
stayed_df = data[data['Attrition'] == 0]
print(f'Total Employees: {len(data)}\n')
print(f'Number of employees who left: {data["Attrition"].value_counts()[1]}')
print(f'% of employees who left: {round(data["Attrition"].value_counts()[1]/len(data)*100,2)}%\n')
print(f'Number of employees who stayed: {data["Attrition"].value_counts()[0]}')
print(f'% of employees who stayed: {round(data["Attrition"].value_counts()[0]/len(data)*100,2)}%')
# Count the number of employees who stayed and left
# It seems that we are dealing with an imbalanced dataset
left_df = data[data['Attrition'] == 1]
stayed_df = data[data['Attrition'] == 0]
stayed_df.describe()
left_df.describe()
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
sns.histplot(data, x='Age', hue='Attrition', kde=True, ax=ax[0])
sns.histplot(data, x='DailyRate', hue='Attrition', kde=True, ax=ax[1])
sns.histplot(data, x='DistanceFromHome', hue='Attrition', kde=True, ax=ax[2])
correlations = data.corr(numeric_only=True, method='spearman') #We're using Spearman's Correlation Coefficient as we are dealing with non-parametric data (not normally distributed)
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlations, annot=True)
# Job level is strongly correlated with total working hours
# Monthly income is strongly correlated with Job level
# Monthly income is strongly correlated with total working hours
# Age is stongly correlated with monthly income
data['Attrition'] = data['Attrition'].astype(str)
plt.figure(figsize=(25, 12))
sns.countplot(x='Age', hue='Attrition', data=data)
plt.show()
plt.figure(figsize=[20,20])
plt.subplot(411)
sns.countplot(x = 'JobRole', hue = 'Attrition', data = data)
plt.subplot(412)
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = data)
plt.subplot(413)
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = data)
plt.subplot(414)
sns.countplot(x = 'JobLevel', hue = 'Attrition', data = data)
plt.figure(figsize=(12,7))
sns.kdeplot(left_df['DistanceFromHome'], label='Employees who left', fill=True, color='r')
sns.kdeplot(stayed_df['DistanceFromHome'], label='Employees who Stayed', fill=True, color='b')
plt.xlabel('Distance From Home')
plt.legend()
from scipy.stats import mannwhitneyu
stats, p = mannwhitneyu(left_df["DistanceFromHome"], stayed_df["DistanceFromHome"])
print(f'p-value: {p}')
plt.figure(figsize=(12,7))
sns.kdeplot(left_df['YearsWithCurrManager'], label='Employees who left', fill=True, color='r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label='Employees who Stayed', fill=True, color='b')
plt.xlabel('Years With Current Manager')
plt.legend()
from scipy.stats import mannwhitneyu
stats, p = mannwhitneyu(left_df["YearsWithCurrManager"], stayed_df["YearsWithCurrManager"])
print(f'p-value: {p}')
plt.figure(figsize=(12,7))
sns.kdeplot(left_df['TotalWorkingYears'], label='Employees who left', fill=True, color='r')
sns.kdeplot(stayed_df['TotalWorkingYears'], label='Employees who Stayed', fill=True, color='b')
plt.xlabel('Total Working Years')
plt.legend()
from scipy.stats import mannwhitneyu
stats, p = mannwhitneyu(left_df["TotalWorkingYears"], stayed_df["TotalWorkingYears"])
print(f'p-value: {p}')
sns.boxplot(x='Gender', y='MonthlyIncome', data=data)
from scipy.stats import mannwhitneyu
male_income = data[data['Gender'] == 'Male']['MonthlyIncome']
female_income = data[data['Gender'] == 'Female']['MonthlyIncome']

stats, p = mannwhitneyu(male_income, female_income)
print(f'p-value: {p}')
plt.figure(figsize=(15, 10))
sns.boxplot(x='MonthlyIncome', y='JobRole', data=data)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
#Separating all categorical data from the dataset
X_cat = data[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_cat = ohe.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)
#assigning column names
X_cat.columns = ohe.get_feature_names_out(['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'])
X_num = data[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 
              'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 
              'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
              'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

X_all = pd.concat([X_num, X_cat], axis=1)
X_cat.shape
X_all
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['Attrition'], test_size=0.25)
#perform a sampling technique to balance the dataset for data['Attrition'] ==1 
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, sampling_strategy='minority')
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
from imblearn.over_sampling import RandomOverSampler
# Perform random oversampling
ros = RandomOverSampler(random_state=0, sampling_strategy='minority')
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
from imblearn.combine import SMOTETomek
# Perform random sampling
smotetomek = SMOTETomek(random_state=0)
X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)
from sklearn.linear_model import LogisticRegression

lreg_model = LogisticRegression()
lreg_model.fit(X_train_smote, y_train_smote)

lreg_model.score(X_test, y_test)

from sklearn.metrics import confusion_matrix, classification_report

y_pred = lreg_model.predict(X_test)
lreg_cnf = confusion_matrix(y_test, y_pred)
sns.heatmap(lreg_cnf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier()
rfc_model.fit(X_train_smote, y_train_smote)

rfc_model.score(X_test, y_test)

y_pred = rfc_model.predict(X_test)
rfc_cnf = confusion_matrix(y_test, y_pred)
sns.heatmap(rfc_cnf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Classifier Confusion Matrix')
plt.show()
print(classification_report(y_test, y_pred))
import xgboost as xgb

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_smote, y_train_smote)

xgb_model.score(X_test, y_test)

y_pred = xgb_model.predict(X_test)
xgb_cnf = confusion_matrix(y_test, y_pred)
sns.heatmap(xgb_cnf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Classifier Confusion Matrix')
plt.show()
print(classification_report(y_test, y_pred))
pip install catboost
cb_model = cb.CatBoostClassifier(verbose=0)
cb_model.fit(X_train_smote, y_train_smote)

cb_model.score(X_test, y_test)

y_pred = cb_model.predict(X_test)
cb_cnf = confusion_matrix(y_test, y_pred)
sns.heatmap(cb_cnf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CatBoost Classifier Confusion Matrix')
plt.show()
print(classification_report(y_test, y_pred))
ann_cnf = confusion_matrix(y_test, y_pred)
sns.heatmap(ann_cnf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Artificial Neural Network Confusion Matrix')
plt.show()
print(classification_report(y_test, y_pred))






