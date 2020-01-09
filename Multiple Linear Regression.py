# -*- coding: utf-8 -*-
#Linear Regression in Python with Scikit-Learn
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
#%matplotlib inline

#--------------IMPORT DATA HERE-----------------------
dataset = pd.read_csv(r'C:\HỌC TẬP\PYTHON LEARNING\STATISTIC\DATA RESEARCH\chỉ số signup.csv')

#tính kích cỡ dữ liệu
print('>>>>>> KÍCH CỠ DỮ LIỆU:',dataset.shape)
print('>>>>>> CÁC BIẾN TRONG DATA:')
print(dataset.dtypes) #xác định các biến có trong data (tìm tiêu đề))

#--------xác định 8 chỉ số cơ bản của biến cụ thể-----
print('>>>>>> 8 CHỈ SỐ THỐNG KÊ CƠ BẢN')
print(dataset['Chỉ số Signup'].describe())

# tổng kết các câu lệnh statistic trong pandas
# 1	count()	Number of non-null observations
# 2	sum()	Sum of values
# 3	mean()	Mean of Values
# 4	median()	Median of Values
# 5	mode()	Mode of values
# 6	std()	Standard Deviation of the Values
# 7	min()	Minimum Value
# 8	max()	Maximum Value
# 9	abs()	Absolute Value
# 10	prod()	Product of Values
# 11	cumsum()	Cumulative Sum
# 12	cumprod()	Cumulative Product

#kiểm tra & loại bỏ dữ liệu Null
dataset.isnull().any()
dataset = dataset.fillna(method='ffill')

#------------ MATRIX HỆ SỐ TƯƠNG QUAN -------------------
#seabornInstance.pairplot(dataset)

#------------ xác định chuỗi biến phụ thuộc -------------
X = dataset[['Số người tiếp cận được', 
             'Số lần hiển thị', 
             'Tần suất', 
#             'Số tiền đã chi tiêu (VND)', 
#             'Số lượt click vào liên kết', 
#             'Số lần hiển thị', 
#             'Chi phí tiếp cận 1.000 người', 
#             'Lượt tương tác với bài viết',
             'Số tiền đã chi tiêu (VND)']].values
y = dataset['Chỉ số Signup'].values

#----------- plot phân phối chuẩn biến độc lập----------
plt.figure(figsize=(5,5))
plt.tight_layout()
seabornInstance.distplot(dataset['Chỉ số Signup'])

#----------------- MATRIX CORR -------------------------
fig, ax = plt.subplots(figsize=(8,6)) 
seabornInstance.heatmap(dataset.corr(), annot=True,vmin=-1, vmax=1,center=0,
           annot_kws={'size': 12},ax=ax)

#Test 20% dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#thiếp lập mô hình
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#chỉ số tính toán từng biến của mô hình
coeff_df = pd.DataFrame(regressor.coef_)  
print(coeff_df)
#----------------------------------------------------------------------

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score
print('>>>>>> R Squared - Độ phù hợp của mô hình:',r2_score(y_test, y_pred))

a=float(((np.sqrt(metrics.mean_squared_error(y_test, y_pred)))/(metrics.mean_absolute_error(y_test, y_pred)))-1)
if a<=0.05:
    print('>>>>>> độ chính xác cao')
    if a<0.1 and a>0.05:
        print('>>>>>> độ chính xác tương đối')
else:
    print('>>>>>> (RMSE/MAE - 1)=',a)