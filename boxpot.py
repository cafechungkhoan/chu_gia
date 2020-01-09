import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
#------------------IMPORT DATA HERE-------------------------
var=pd.read_csv(r'C:\PYTHON LEARNING\STATISTIC\DATA RESEARCH\abs video.csv')
var.head()
var2=DataFrame(var)

#------------- BOX PLOT --------------------------
var2.drop('Số lượt click vào liên kết', axis=1).plot(kind='box', 
         subplots=True, sharex=False, sharey=False, figsize=(40,7), 
         title='--- BOXPLOT EACH VAR ---')
plt.show()

#‘bar’ or ‘barh’ for bar plots
#‘hist’ for histogram
#‘box’ for boxplot
#‘kde’ or 'density' for density plots
#‘area’ for area plots
#‘scatter’ for scatter plots
#‘hexbin’ for hexagonal bin plots
#‘pie’ for pie plots