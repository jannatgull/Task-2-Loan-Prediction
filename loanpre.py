import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#load the dataset
df = pd.read_csv('loanpred.csv' , delimiter= ',' )
print(df)
#Basic data summarization
#displaying the structure of dataset
print(df.shape)
#displaying first 5 rows of dataset
print(df.head())
#displaying the names of columns of dataset
print(df.columns)
"""Missing data
There is no missing data in this dataset"""
#Visualization of data 
#Histogram is used to visualize the distribution of Annual Salary
sns.set(style= 'whitegrid')
plt.figure(figsize=(8,5))
sns.histplot(data= df , x = 'Annual Salary')
plt.title("Annual Salary Distribution")
plt.show()
read = input("Wait for me...")
#Box plot is used to visualize the distribution of Bank Balance
sns.set(style= 'whitegrid')
plt.figure(figsize=(8,5))
sns.boxplot(data= df, x= 'Bank Balance')
plt.title("Bank Balance Distribution")
plt.show()
read = input("Wait for me...")
#Count plot is used to visualize the distribution of Employment status
sns.set(style= 'whitegrid' )
plt.figure(figsize= (8,5))
sns.countplot(data= df, x= 'Employed')
plt.title("Employment Status Distribution")
plt.show()
read = input("Wait for me...")
#Seperating X and Y for model training
X = df[['Index','Employed','Bank Balance','Annual Salary']]
Y = df['Defaulted?']
#Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , Y , train_size= 0.8 , random_state= 42)
#Training the Logistic Regression classification model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train , y_train)
#Predicting the target values using the trained Logistic Regression model
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score , confusion_matrix
#Calculating the accuracy of the trained model
acc = accuracy_score(y_test , y_pred)
print("Accuracy score:" , round(acc , 3))
#Creating a confusion matrix to analyze correct and incorrect predictions
cr = confusion_matrix(y_test , y_pred)
print("Classification matrix:" , cr)
#Training the decision tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train , y_train)
#Predicting the target values using the trained Decision tree model
y_pred = model.predict(x_test)
#Calculating the accuracy of the trained model
acc = accuracy_score(y_test , y_pred)
print("Accuracy score:" , round(acc , 3))
#Creating a confusion matrix to analyze correct and incorrect predictions
cr = confusion_matrix(y_test , y_pred)
print("Classification matrix:" , cr)