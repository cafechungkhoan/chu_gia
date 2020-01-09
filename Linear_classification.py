import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\PYTHON LEARNING\STATISTIC\DATA RESEARCH\khách hàng tiềm năng.csv')
data.head()
import seaborn as sns
from pandas import DataFrame

#định danh yếu tố chủ chốt
#sns.countplot(data['Khách hàng tiềm năng'],label="Count")
#plt.show()

#----------- MATRIC SCATTER ---------------------------------------
sns.pairplot(data,diag_kind="kde", markers="+")

#---------- MATRIX CORR -------------------------------------------
var2=DataFrame(data)
fig, ax = plt.subplots(figsize=(10,8)) 
sns.heatmap(var2.corr(), annot=True,vmin=-1, vmax=1,center=0,
           annot_kws={'size': 10},ax=ax)


#Trực quan trên biểu đồ BOXPLOT, định danh tiêu đề
data.drop('Khách hàng tiềm năng', axis=1).plot(kind='box', 
         subplots=True, sharex=False, sharey=False, figsize=(33,7), 
         title='--- BOXPLOT TỪNG BIẾN TRÊN KH TIỀM NĂNG ---')
plt.show()

#Histogram for each numeric
#import pylab as pl
#data.drop('Khách hàng tiềm năng' ,axis=1).hist(bins=60, figsize=(12,10))
#pl.suptitle("--- HISTOGRAM FOR MAIN NUMERIC ---")
#plt.show()
sns.jointplot(x=data['Số người tiếp cận được'], 
              y=data['Khách hàng tiềm năng'],
                  kind="kde", height=7, space=0,color="darkviolet")

sns.jointplot(x=data['Lượt tương tác với bài viết'], 
              y=data['Khách hàng tiềm năng'],
                  kind="kde", height=7, space=0,color="darkviolet")

sns.jointplot(x=data['Lượt phát video trong tối thiểu 3 giây'], 
              y=data['Khách hàng tiềm năng'],
                  kind="kde", height=7, 
                  space=0,
                  color="darkviolet").plot_joint(plt.scatter,
                                    c='w',s=30,linewidth=1,marker='+')

