#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Create a dataframe
iris = pd.read_csv("IRIS.csv")


# In[3]:


iris.head(10)


# In[4]:


iris.shape


# In[5]:


iris.info()


# In[6]:


# Checking null values
iris.isna().sum()


# In[7]:


#Checking duplicate values
iris.duplicated()


# In[8]:


iris.duplicated().sum()


# In[9]:


# Remove duplicate values
iris= iris.drop_duplicates()


# In[10]:


iris.shape


# In[11]:


iris


# In[12]:


#Outliers detection
fig,ax = plt.subplots(figsize=(10,5))
sns.boxplot(iris,ax=ax)


# In[13]:


#Summary of the data
iris.describe()


# In[14]:


iris.species.unique()


# In[18]:


iris['species'].value_counts()


# In[19]:


# Plotting the species data
sns.countplot(x = "species",data=iris,palette="Dark2")


# In[20]:


# Histogram plots for numerical features
fig,ax=plt.subplots(2,2, figsize=(12,8))

ax[0,0].hist(iris['sepal_length'],bins=20,edgecolor='black')
ax[0,0].set_xlabel('sepal_length')
ax[0,0].set_ylabel('frequency')

ax[0,1].hist(iris['sepal_width'],bins=20,edgecolor='black',color='brown')
ax[0,1].set_xlabel('sepal_width')
ax[0,1].set_ylabel('frequency')

ax[1,0].hist(iris['petal_length'],bins=20,edgecolor='black',color='purple')
ax[1,0].set_xlabel('petal_length')
ax[1,0].set_ylabel('frequency')

ax[1,1].hist(iris['petal_width'],bins=20,edgecolor='black',color='green')
ax[1,1].set_xlabel('petal_width')
ax[1,1].set_ylabel('frequency')
plt.show()


# In[21]:


# Checking relationship between variables
fig,ax=plt.subplots()
sns.heatmap(iris.drop(columns='species').corr(), annot=True,cmap='Spectral',fmt=".2f",ax=ax,linewidth=0.5)
plt.show()


# In[22]:


# Scatterplot for petal length and petal width by species
fig,ax=plt.subplots(figsize=(8,6))
sns.scatterplot(x="petal_width",y="petal_length",hue="species",data=iris)
plt.show()


# In[23]:


# Scatterplot for sepal length and sepal width by species
fig,ax=plt.subplots(figsize=(8,6))
sns.scatterplot(x="sepal_width",y="sepal_length",hue="species",data=iris)
plt.show()


# In[24]:


# Scatterplot for sepal length and petal length by species
fig,ax=plt.subplots(figsize=(8,6))
sns.scatterplot(x="petal_length",y="sepal_length",hue="species",data=iris)
plt.show()


# In[25]:


#Pairplot for data
sns.pairplot(iris, hue='species')


# In[26]:


# Label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# In[27]:


iris['species']=label_encoder.fit_transform(iris['species'])


# In[28]:


iris


# In[29]:


from sklearn.model_selection import train_test_split

# Define feature and target variables
x= iris.drop('species',axis=1)
y=iris['species']

#Split dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)

train_data= x_train.join(y_train)
train_data


# In[30]:


from sklearn.linear_model import LogisticRegression

#Build a Logisitc Regression model
fitted_model_lr = LogisticRegression()
#Train the model
fitted_model_lr.fit(x_train,y_train)
#Make predictions
y_pred_lr = fitted_model_lr.predict(x_test)


# In[31]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model's performance evaluating
accuracy = accuracy_score(y_test, y_pred_lr)
print(f' Accuracy for LR: {accuracy:.4f}')
print(classification_report(y_test, y_pred_lr))
cf=confusion_matrix(y_test,y_pred_lr)
sns.heatmap(cf,annot=True,fmt=".2f",cmap="Spectral",linewidth=0.5)
plt.title('Confusion Matrix for LR Model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


# In[32]:


from sklearn.neighbors import KNeighborsClassifier

#Build a KNN model
fitted_model_knn = KNeighborsClassifier()
#Train the model
fitted_model_knn.fit(x_train,y_train)
#Make predictions
y_pred_knn = fitted_model_knn.predict(x_test)


# In[33]:


# Model's performance evaluating
accuracy1 = accuracy_score(y_test, y_pred_knn)
print(f' Accuracy for KNN: {accuracy1:.4f}')
print(classification_report(y_test, y_pred_knn))
cf1=confusion_matrix(y_test,y_pred_knn)
sns.heatmap(cf1,annot=True,fmt=".2f",cmap="Spectral",linewidth=0.5)
plt.title('Confusion Matrix for KNN Model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


# In[34]:


from sklearn.svm import SVC

#Build a KNN model
fitted_model_svm = SVC()
#Train the model
fitted_model_svm.fit(x_train,y_train)
#Make predictions
y_pred_svm = fitted_model_svm.predict(x_test)


# In[35]:


# Model's performance evaluating
accuracy2 = accuracy_score(y_test, y_pred_svm)
print(f' Accuracy for SVM: {accuracy2:.4f}')
print(classification_report(y_test, y_pred_svm))
cf2=confusion_matrix(y_test,y_pred_svm)
sns.heatmap(cf2,annot=True,fmt=".2f",cmap="Spectral",linewidth=0.5)
plt.title('Confusion Matrix for SVM Model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

