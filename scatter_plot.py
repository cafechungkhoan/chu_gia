# vẽ đường hồi quy tuyến tính
#Linear Regression in Python with Scikit-Learn
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

#-------------- IMPORT DATA HERE --------------
dataset = pd.read_csv(r'C:\PYTHON LEARNING\research signup.csv')

#thay biến vào vị trí
dataset.plot(x='Tần suất', y='Khách hàng tiềm năng', style='o')
#------------------ đặt tên tiêu đề-----------------  

#plt.title('--- LINEAR REGRESSION ---')
plt.scatter('Tần suất', 'Khách hàng tiềm năng',  color='gray')
plt.plot('Tần suất', 'Khách hàng tiềm năng', color='red', linewidth=2)
plt.xlabel('Tần suất')  
plt.ylabel('Khách hàng tiềm năng')  
plt.show()