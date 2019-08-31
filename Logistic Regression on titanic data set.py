# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:11:41 2019

@author: Dilip
"""

#collect data: import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
#%matplotlib inline

#import data set
titanic_data=pd.read_csv('titanic-data.csv')
print("# of passengers in original data:"+str(len(titanic_data.index)))
      
#analyzing data
sns.countplot(x="Survived",data=titanic_data)
sns.countplot(x="Survived",hue="Sex",data=titanic_data)
sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist()
titanic_data.info()
sns.countplot(x="SibSp",data=titanic_data)


#Data Wrangling
titanic_data.isnull()
titanic_data.isnull().sum()

sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap="viridis")

sns.boxplot(x="Pclass",y="Age",data=titanic_data)

titanic_data.head(10)

titanic_data.drop("Cabin",axis=1,inplace=True)
titanic_data.head(10)

titanic_data.dropna(inplace=True)

sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap="viridis")

titanic_data.isnull().sum()

titanic_data.head(10)

#creating dummies variable
sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
sex.head(5)
embarked=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embarked.head(5)
pcls=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
embarked.head(5)

#concat all new rows in a the data set
newtitanic_data=pd.concat([titanic_data,sex,embarked,pcls],axis=1)
newtitanic_data.head(10)

#drop unwanted coloums
newtitanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
newtitanic_data.head(10)

newtitanic_data.drop(['Pclass'],axis=1,inplace=True)
newtitanic_data.head(10)

#split the  data set
x=newtitanic_data.drop("Survived",axis=1)
y=newtitanic_data["Survived"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#import libreries
from sklearn.linear_model import LogisticRegression
l_regression=LogisticRegression()
l_regression.fit(x_train,y_train)
y_predict=l_regression.predict(x_test)


from sklearn.metrics import classification_report
classification_report(y_test,y_predict)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predict)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)






