# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:56:06 2017

@author: Aditya Dubey
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



mask = np.array([False,False,True,False,True,True,True,True,False,True,False,True])

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset['Age'].fillna(dataset['Age'].mode()[0],inplace=True)
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
X = dataset.iloc[:, mask].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])


mask1 = np.array([True,True,True,True,True,True,True])
X = X[: , mask1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
#classifier.add(Dropout(p=0.05)) #used for preventing overfitting
# Adding the second hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p=0.05))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train,y_train, batch_size = 10 , epochs = 100)
# Part 3 - Making predictions and evaluating the mode

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print("  ".format(y_pred,y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





