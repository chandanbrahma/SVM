# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:30:03 2020

@author: chandan
"""

### importing the dataset
import pandas as pd
import numpy as np
data1= pd.read_csv('E:\\assignment\\svm\\SalaryData_Test(1).csv')

data2= pd.read_csv('E:\\assignment\\svm\\SalaryData_Train(1).csv')

##combining data1 and data2
frame=[data1,data2]
data = pd.concat(frame)


data.head()
data.describe()
data.info()
## in total we do have 45221 rows and 14 columns, also we donot have any null values
## However inside our dataset we do have some catagorical columns thoose need to be converted into int format

data['workclass'],_= pd.factorize(data.workclass)

data['education'],_= pd.factorize(data.education)

data['maritalstatus'],_= pd.factorize(data.maritalstatus)

data['occupation'],_= pd.factorize(data.occupation)

data['relationship'],_ = pd.factorize(data.relationship)

data['race'],_ = pd.factorize(data.race)

data['sex'],_ = pd.factorize(data.sex)

data['native'],_ = pd.factorize(data.native)

data['Salary'],_ = pd.factorize(data.Salary)

data.head()
data.info()


##normalizing the dataset
data_new= (data-data.min())/(data.max()-data.min())

data_new.describe()



###model building
## segrigating the predictors and the target variables
x= data_new.iloc[:,:-1]
y= data_new.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


##importing support vector machine and fitting it to the data
from sklearn.svm import SVC

## kernal= linear
model2 = SVC(kernel = "linear")
model2.fit(x_train, y_train)
y_pred= model2.predict(x_test)
np.mean(y_pred==y_test) ##80.54
## we got an accuracy of 80.54 % so lets also try with poly and rbf kernel

##kernel= poly
model = SVC(kernel = "poly")
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
np.mean(y_pred==y_test) ##84.07


##kernel= rbf
model1 = SVC(kernel = "rbf")
model1.fit(x_train, y_train)
y_pred= model1.predict(x_test)
np.mean(y_pred==y_test) ##84.23

## so in rbf kernel we got the highest accuracy of 84.23%.
