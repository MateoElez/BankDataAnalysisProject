#Ucitvanaje potrebnih biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#Interface za prikaz podataka
sns.set()

#Učitavanje podataka
data = pd.read_csv('arrangedData.csv')
print(data.shape)
print(data.describe())

## Vadimo potrebne varijable
# Regresori X su sve varijable osim 'subscribed', y je 'subscribed'

print('*****************Logistička regresija 1.*****************************')

X = data.loc[:,data.columns != 'subscribed'].copy()
y = data.loc[:,data.columns == 'subscribed'].copy()

""" X = data.iloc[:, 0:14].copy().values
y = data.iloc[:, 14].copy().values """


#Dijelimo podatke na trenirane i testne, omjer je 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


#Provodimo logističku regresiju
classifier= LogisticRegression()  
classifier.fit(X_train, y_train)  

#Predviđamo rezultat 
y_pred= classifier.predict(X_test) 

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("Score for logistic regression model is: ", classifier.score(X_test, y_test))
print("Classification report table:")
print(classification_report(y_test, y_pred))

## Vadimo potrebne varijable
# Regresori X su sve varijable osim 'subscribed', y je 'subscribed'

print('*****************Logistička regresija 2.*****************************')

print("Pretvaranje varijable housing iz numeric u kategorijsku")
data['housing'] = pd.Categorical(data.housing)

import warnings
warnings.filterwarnings('always')

X = data.loc[:,['age', 'marital', 'education']].copy()
y = data.loc[:,['housing']].copy()


#Dijelimo podatke na trenirane i testne, omjer je 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#Provodimo logističku regresiju
classifier= LogisticRegression()  
classifier.fit(X_train, y_train)  

#Predviđamo rezultat 
y_pred= classifier.predict(X_test) 

#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
print("Score for logistic regression model is: ", classifier.score(X_test, y_test))
print("Classification report table:")
print(classification_report(y_test, y_pred))



print('*****************Logistička regresija 3.*****************************')

print("Pretvaranje varijable housing iz numeric u kategorijsku")
data['loan'] = pd.Categorical(data.loan)

import warnings
warnings.filterwarnings('always')

X = data.loc[:,['age', 'marital', 'education']].copy()
y = data.loc[:,['loan']].copy()


#Dijelimo podatke na trenirane i testne, omjer je 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#Provodimo logističku regresiju
classifier= LogisticRegression()  
classifier.fit(X_train, y_train)  

#Predviđamo rezultat 
y_pred= classifier.predict(X_test) 

#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
print("Score for logistic regression model is: ", classifier.score(X_test, y_test))
print("Classification report table:")
print(classification_report(y_test, y_pred))


print('*****************Logistička regresija 4.*****************************')

X = data.loc[:,['age', 'marital', 'education']].copy()
y = data.loc[:,['subscribed']].copy()


#Dijelimo podatke na trenirane i testne, omjer je 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#Provodimo logističku regresiju
classifier= LogisticRegression()  
classifier.fit(X_train, y_train)  

#Predviđamo rezultat 
y_pred= classifier.predict(X_test) 

#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
print("Score for logistic regression model is: ", classifier.score(X_test, y_test))
print("Classification report table:")
print(classification_report(y_test, y_pred))


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


import seaborn as sns
sns.set(style="ticks")

sns.pairplot(data, hue="education", palette="Set1")
plt.show()

sns.pairplot(data, hue="marital", palette="Set1")
plt.show()

sns.pairplot(data, hue="age", palette="Set1")
plt.show()


