#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:34:58 2017

@author: aditya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('train.csv')
Y=dataset.iloc[:20000,4].values

from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
Y=lc.fit_transform(Y)

import re
import nltk as nl
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()
corpus=[]
for i in range(0,20000):
    review=dataset['Description'][i]
    review=re.sub('[^a-zA-Z0-9]',' ',review)
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(words )for words in review if not words in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    print("i=====>",i)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

import keras 
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(output_dim=12000,init='uniform',activation='relu',input_dim=23343))
classifier.add(Dense(output_dim=1000,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=500,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=32,epochs=100)
pred=classifier.predict(X_test)
pred=pred>0.5
pred1=classifier.predict_proba(X_test)

    

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,pred)
