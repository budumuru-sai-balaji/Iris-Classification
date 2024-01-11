import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def LogRegression(a,b,c,d):
  data  = pd.read_csv("../IRIS.csv")
  x = data.drop(columns=["species"])
  y = data['species']

  X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20)



  LogReg = LogisticRegression()
  LogReg.fit(X_train,y_train)
  predInp = pd.DataFrame(np.array([[a,b,c,d]]),columns=['sepal_length','sepal_width','petal_length','petal_width'])

  y_pred = LogReg.predict(predInp)

  print(y_pred[0])


try:
  sl = float(input("Enter the Sepal Length: "))
  sw = float(input("Enter the Sepal width: "))
  pl = float(input("Enter the Petal Length: "))
  pw = float(input("Enter the Petal width: "))
  ls=[sl,sw,pl,pw]
  LogRegression(sl,sw,pl,pw)

except ValueError:
  print("Invalid input!!!")


