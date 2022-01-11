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

#pretvaranje varijable housing iz numeric u kategorijsku..
print("Pretvaranje varijable housing iz numeric u kategorijsku")
data['housing'] = pd.Categorical(data.housing)
print(data.dtypes)


#za LOGISTICKU REGRESIJU
# X5 je age, a y4 je housing
X5 = data.loc[:,['age']].copy()
y5 = data.loc[:,['housing']].copy()

# lab_enc = preprocessing.LabelEncoder()
# age = lab_enc.fit_transform(X5)

#Dijelimo podatke na trenirane i testne, omjer je 70:30
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.30, random_state=0)


model = LogisticRegression()

#Logistic regression for age and housing

print("Logistic regression for age and housing")
print(model.fit(X5_train, y5_train))

print("Score for logistic regression model is: ", model.score(X5_test, y5_test))

y5_pred = model.predict(X5_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y5_test, y5_pred)
print(confusion_matrix)

print(classification_report(y5_test, y5_pred))


#za LOGISTICKU REGRESIJU2
# X5 je subscription, a y4 je ostalo
X5 = data.loc[:,data.columns != 'subscribed'].copy()
y5 = data.loc[:,data.columns == 'subscribed'].copy()

# lab_enc = preprocessing.LabelEncoder()
# age = lab_enc.fit_transform(X5)

#Dijelimo podatke na trenirane i testne, omjer je 70:30
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.30, random_state=0)


model = LogisticRegression()

#Logistic regression for age and housing

print("Logistic regression for age and housing")
print(model.fit(X5_train, y5_train))

print("Score for logistic regression model is: ", model.score(X5_test, y5_test))

y5_pred = model.predict(X5_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y5_test, y5_pred)
print(confusion_matrix)

print(classification_report(y5_test, y5_pred))

#Logistic regression for CPI and loan
X6 = data.loc[:,['cons.price.idx']].copy()
y6 = data.loc[:,['loan']].copy()

# lab_enc = preprocessing.LabelEncoder()
# age = lab_enc.fit_transform(X5)

#Dijelimo podatke na trenirane i testne, omjer je 70:30
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.30, random_state=0)

print("Logistic regression for age and housing")
model.fit(X6_train, y6_train)

print("Score for logistic regression model is: ", model.score(X6_test, y6_test))

#Logistic regression for CFI and housing
X7 = data.loc[:,['cons.conf.idx']].copy()
y7 = data.loc[:,['housing']].copy()

# lab_enc = preprocessing.LabelEncoder()
# age = lab_enc.fit_transform(X5)

#Dijelimo podatke na trenirane i testne, omjer je 70:30
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.30, random_state=0)

print("Logistic regression for CFI and housing")
model.fit(X7_train, y7_train)

print("Score for logistic regression model is: ", model.score(X7_test, y7_test))