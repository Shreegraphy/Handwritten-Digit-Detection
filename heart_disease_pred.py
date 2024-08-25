import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
heartdata=pd.read_csv('heart_disease_data.csv')
heartdata.shape
heartdata.isnull().sum()
heartdata.info()
x=heartdata.drop('target',axis=1)
y=heartdata['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
x_train_pred=model.predict(x_train)
x_train_acu=accuracy_score(x_train_pred,y_train)
print(x_train_acu)
x_test_pred=model.predict(x_test)
x_test_acu=accuracy_score(x_test_pred,y_test)
print(x_test_acu)
input_data = (58,0,3,150,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.array(input_data)
print(input_data_as_numpy_array)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
