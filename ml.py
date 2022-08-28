#SLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

data_set = pd.read_csv("C:/Users/bhakt/.spyder-py3/Salary_Data.csv")
print(data_set)

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,1].values

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 10,train_size = 20, random_state=0)  

from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)

pred_train = regressor.predict(x_train)

mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, pred_train, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()   


#test
pred_test = regressor.predict(x_test)

#visualizing the Test set results  
mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, pred_train, color="red")    
mtp.title("Salary vs Experience (Test Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  

