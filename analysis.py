from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
      if request.method=='POST':
           Pregnancies=float(request.form['pregnancies'])
           Glucose=float(request.form['glucose'])
           BloodPressure=float(request.form['bloodPressure'])
           SkinThickness=float(request.form['skinThickness'])
           Insulin=float(request.form['insulin'])
           BMI=float(request.form['BMI'])
           DiabetesPedigreeFunction=float(request.form['diabetesPedigreeFunction'])
           Age=float(request.form['age'])

           data= pd.read_csv("D:\diabProject/diabetes.csv")
           
           #splitting data
           X=data.drop("Outcome",axis=1)
           Y=data["Outcome"]
           X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

            #training he model
           model=LogisticRegression()
           model.fit(X_train,Y_train)

           #prediction
           prediction=model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

           result=""
           if prediction==[0]:
                result="Negatif"
           elif prediction==[1]:
                result="Positive"   
           return render_template('index.html', result=result)
      else:
           result=""
           return render_template('index.html',result=result)



if __name__ =="__main__":
    app.run(debug=True)           


