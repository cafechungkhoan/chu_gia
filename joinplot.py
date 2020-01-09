import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\HỌC TẬP\PYTHON LEARNING\STATISTIC\DATA RESEARCH\quản lí chi phí.csv')
data.head()
import seaborn as sns
from pandas import DataFrame

sns.jointplot(x=data['CPM (Chi phí trên mỗi 1.000 lần hiển thị)'], 
              y=data['Khách hàng tiềm năng'],
                  kind="kde", height=7, space=0,color="darkviolet").plot_joint(plt.scatter,
                                    c='w',s=30,linewidth=1,marker='+')

sns.jointplot(x=data['CPC (Chi phí trên mỗi lượt click vào liên kết)'], 
              y=data['Khách hàng tiềm năng'],
                  kind="kde", height=7, space=0,color="darkviolet").plot_joint(plt.scatter,
                                    c='w',s=30,linewidth=1,marker='+')

sns.jointplot(x=data['Số người tiếp cận được'], 
              y=data['Khách hàng tiềm năng'],
                  kind="kde", height=7, 
                  space=0,
                  color="darkviolet").plot_joint(plt.scatter,
                                    c='w',s=30,linewidth=1,marker='+')