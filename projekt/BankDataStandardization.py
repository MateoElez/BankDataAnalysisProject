#Ucitvanaje potrebnih biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
ss = StandardScaler()
data_scaled = pd.DataFrame(columns=data.columns, data=ss.fit_transform(data))
print('Standardizirani podatci.')
print(data_scaled)

#Podjela stupaca u x i y varijable
#X = data.iloc[:, :-1].values
#y = data.iloc[:, 4].values

#Klastere radimo u 3 grupe
# X1 je age, job, marital , a y1 je housing
X1 = data_scaled.iloc[:, 0:3].copy().values
y1 = data_scaled.iloc[:, 5:6].copy().values

# X2 je age, job, marital, housing , a y2 je housing
X2 = data_scaled.loc[:,['age', 'job', 'marital', 'housing']].copy().values
y2 = data_scaled.loc[:,['cons.price.idx']].copy().values

# X2 je age, job, marital, housing , a y2 je housing
X3 = data_scaled.loc[:,['age', 'job', 'marital', 'loan']].copy().values
y3 = data_scaled.loc[:,['cons.conf.idx']].copy().values


print(X2)
print(y2)