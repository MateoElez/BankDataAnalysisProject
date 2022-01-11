#Ucitvanaje potrebnih biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#Interface za prikaz podataka
sns.set()

#Učitavanje podataka
data = pd.read_csv('arrangedData.csv')
print(data.shape)
print(data.describe())

#Ispitivanje sredivanja podataka, jesu li ostali null podatci i slicno
for col in data.columns:
    print('{} : {}'.format(col,data[col].unique()))

data.info()

# Potrebno je skalirati podatke tako da svi budu otprilike iste dimenzije
# sklearn radi loše sa kategoričkim podatcima
# Ova stand ne radi bas
ss = StandardScaler()
data_scaled = pd.DataFrame(columns=data.columns, data=ss.fit_transform(data))
print('Standardizirani podatci.')
print(data_scaled)

#Standard broj 2.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#data_scaled.to_csv('bankDataScaled.csv', index = False)

#Podjela stupaca u x i y varijable

#Klastere radimo u 3 grupe
# X1 je age, job, marital , a y1 je housing
X1 = data.iloc[:, 0:3].copy().values
y1 = data.iloc[:, 5:6].copy().values

# X2 je age, job, marital, housing , a y2 je CPI
X2 = data_scaled.loc[:,['age', 'job', 'marital', 'housing']].copy().values
y2 = data_scaled.loc[:,['cons.price.idx']].copy().values

# X3 je age, job, marital, loan , a y3 je Consumer Confidence Index
X3 = data_scaled.loc[:,['age', 'job', 'marital', 'loan']].copy().values
y3 = data_scaled.loc[:,['cons.conf.idx']].copy().values

# X4 je age, a y4 je CPI
X4 = data_scaled.loc[:,['education']].copy().values
y4 = data_scaled.loc[:,['cons.price.idx']].copy().values

# X5 je age, a y4 je CPI
X5 = data_scaled.loc[:,['age']].copy().values
y5 = data_scaled.loc[:,['housing']].copy().values

lab_enc = preprocessing.LabelEncoder()
age = lab_enc.fit_transform(X5)

#Dijelimo podatke na trenirane i testne, omjer je 70:30
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.30)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.30)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.30)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.30)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.30)


clf = LinearRegression()

#Linear regrerssion for 2
print("Linear regression 2\n")
clf.fit(X2_train, y2_train)
print("Predict: \n")
print(clf.predict(X2_test))

print("Score 2 is: ", clf.score(X2_test, y2_test))

#Linear regrerssion for 3
print("Linear regression 3\n")
clf.fit(X3_train, y3_train)
print("Predict: \n")
print(clf.predict(X3_test))

print("Score 3 is: ", clf.score(X3_test, y3_test))

#Linear regrerssion for 4
print("Linear regression 4\n")
clf.fit(X4_train, y4_train)
print("Predict: \n")
print(clf.predict(X4_test))

print("Score 4 is: ", clf.score(X4_test, y4_test))

#Linear regrerssion for X1 and y1
print("Linear regression 1\n")

clf.fit(X1_train, y1_train)
print("Predict: \n")
print(clf.predict(X1_test))

print("Score 1 is: ", clf.score(X1_test, y1_test))
