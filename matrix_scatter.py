import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(style="ticks")
var=pd.read_csv(r'C:\HỌC TẬP\PYTHON LEARNING\chỉ số signup.csv')
var.head()
var2=pd.DataFrame(var)
sns.pairplot(var2)