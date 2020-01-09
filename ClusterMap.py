# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
#------------------IMPORT DATA HERE-------------------------
var=pd.read_csv(r'C:\HỌC TẬP\PYTHON LEARNING\STATISTIC\DATA RESEARCH\abs video.csv')
var.head()
var2=DataFrame(var)
#-----------------PLOT CORR MATRIX--------------------------
import seaborn as sn
fig, ax = plt.subplots(figsize=(12,10)) 
a=sn.heatmap(var2.corr(), annot=True,vmin=-1, vmax=1,center=0,
           annot_kws={'size': 9},ax=ax)
plt.title('----------- Matrix Corr ------------')
#var2.describe()
#-----------------PLOT CORR CLUSTER ------------------------
b=sn.clustermap(var2.corr(), annot=True,vmin=-1, vmax=1,center=0,
           annot_kws={'size': 9})
plt.title('-------------- Matrix Corr Clustermap --------------')
#------------ convert thành ảnh ----------------------
#b.savefig('plot.png')