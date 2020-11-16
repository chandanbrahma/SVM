# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:07:55 2020

@author: admin
"""
### importing the dataset
import pandas as pd
import numpy as np
data = pd.read_csv('E:\\assignment\\svm\\forestfires.csv')


data.head()

data.describe()

data.info()

##so we do have our dataset with517 rows and 31 columns, We do not have any null values
##but we have 3 columns which need to  be converted into integer format


## converting the catagorical variable into integer format
data.month,_ = pd.factorize(data.month)

data.day,_ = pd.factorize(data.day)

data['size_category'],_ = pd.factorize(data['size_category'])


data.head()
data.info()


##building model
##selecting the prectors and the target variables

x= data.iloc[:,:-1]
y= data.iloc[:,-1]

##training and testing dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

##importing svm and fitting into the data
from sklearn.svm import SVC

## fitting different kernels
##linear kernel
model = SVC(kernel = 'linear')
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
np.mean(y_pred==y_test)
 ## so we got an accuracy of 98.07 % , lets try for rbf kernel


##rbf kernel 
model1 = SVC(kernel = 'rbf')
model1.fit(x_train, y_train)
y_pred= model1.predict(x_test)
np.mean(y_pred==y_test)
## as the acuracy decreased to 77.8%, so linear kernel is the best for the dataset. 