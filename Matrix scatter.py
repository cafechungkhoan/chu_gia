# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
#------------------IMPORT DATA HERE-------------------------
var=pd.read_csv(r'C:\HỌC TẬP\PYTHON LEARNING\STATISTIC\DATA RESEARCH\dell.csv')
var.head()
var2=DataFrame(var)
#-----------------PLOT CORR MATRIX--------------------------
import seaborn as sn
sns.pairplot(var2)


