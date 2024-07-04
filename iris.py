# CLASSIFICATION OF THE SPECIES OF THE IRIS FLOWER.
# There are mainly three types of species iris flower which are following : 
# 1). Setosa , 2). Verticolor , 3). Verginica

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
# loading dataset first
data = load_iris()
x = data.data                               # storing features of iris flowers in x variable.
y = data.target                             # storing target data of iris flowers in x variable. 

df = pd.DataFrame(x , columns = data.feature_names)         # making our data as dataframe

df['species'] = y                          # making new column in our data set named as species which store the target value.

# sns.pairplot(data = df , hue = 'species' )
# plt.show()

# as the target value in 0 ,1 ,2 forms which are 'setosa' , 'verticolor' , 'verginica' simultanously. so we are replacing it.
df['species'] = df['species'].replace(to_replace=[0,1,2] , value=['setosa'  , 'verticolor' , 'verginica'])


# checking if there is any null value in our data set.
# print(df.isnull().sum())

# now for training the model split the data into training and testing dataset.

# we will train our model with 70% of data set , remain 30% data set will be splited as test data set.
x_train , x_test , y_train , y_test = train_test_split(x , y , random_state=42 , test_size= 0.3)

# now make the model.

from sklearn.svm import SVC

# now find the best hyperparameter for our model .
# for that we will use gridsearch cv .

from sklearn.model_selection import GridSearchCV

params = {
    'gamma' : ['auto' , 'scale'],
    'C' : [1 , 10 , 20],
    'kernel' : ['rbf' , 'poly' , 'linear']
}
gd = GridSearchCV(SVC() , params ,cv =3 , return_train_score=False)
gd.fit(x_train , y_train)

gd1 = pd.DataFrame(gd.cv_results_)
# print(gd1)
# print(gd1[['param_C' , 'param_kernel' ,'param_gamma','mean_test_score']])

# print(gd.best_params_)                # {'C': 1, 'gamma': 'auto', 'kernel': 'poly'}
# print(gd.best_score_)                  # 0.952380952

# now we found best parameters for our model

# now make another model and proceed.

sv = SVC(kernel='poly' , gamma='auto' , C=1)

# fit the data and train the model.
sv.fit(x_train , y_train)

# make prdictions by testing data.
predicted = sv.predict(x_test)


# check accuracy by testing.
from sklearn.metrics import accuracy_score

score  = accuracy_score(predicted , y_test)
# print(score)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test , predicted))