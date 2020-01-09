import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
df['target'] = iris.target
df.head()
df[df.target==1].head()
df[df.target==2].head()
df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()
#--------- QUY ĐỊNH VỊ TRÍ PHÂN LỚP ---------------------------
#df[45:55]
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

import matplotlib.pyplot as plt
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], 
                df0['sepal width (cm)'],
                color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], 
                df1['sepal width (cm)'],
                color="blue",marker='.')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], 
                df0['petal width (cm)'],
                color="green",marker='+')
plt.scatter(df1['petal length (cm)'], 
                df1['petal width (cm)'],
                color="blue",marker='.')

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train)
len(X_test)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.score(X_test, y_test)
