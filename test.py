
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

 data= pd.read_csv("D:\diabProject/diabetes.csv")
#print(data)

 #print(sns.heatmap(data.isnull()))

#correlation matrix
#correlation=data.corr()
#print(correlation)

#splitting data
X=data.drop("Outcome",axis=1)
Y=data["Outcome"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
'''In X all the independent variables are stored 
In Y the predictor variable(“OUTCOME”) is stored.
Train-test split is a technique used in machine learning to assess model performance. It 
divides the dataset into a training set and a testing set, with a 0.2 test size indicating that 
20% of the data is used for testing and 80% for training.'''

#training he model
model=LogisticRegression()
model.fit(X_train,Y_train)

#prediction
prediction=model.predict(X_test)
#print(prediction)

accuracy=accuracy_score(prediction,Y_test)
print(accuracy)



