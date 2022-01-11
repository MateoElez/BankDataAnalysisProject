#Ucitvanaje potrebnih biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Interface za prikaz podataka
sns.set()

#Učitavanje podataka
data = pd.read_csv('bank_marketing_dataset.csv')
print(data.shape)
print(data)

#Prikaz tipa podataka stupaca
data.info()

# U ovom odlomku ćemo urediti podatke, razlog za pojedinu izmjenu će 
# biti objašnjeno komentarom
columns = list(data.columns)

for col in columns:
    for i in range(data.shape[0]):
        # U stupcu 'job' ukoliko je nepozant podatak pretpostavljamo da je osoba nezaposlena
        if col == 'job' and data[col][i] == 'unknown':
            data[col][i] = 'unemployed'
        # U stupcu 'marital' ukoliko je nepoznat podatak pretpostavljamo da je osoba sama
        if col == 'marital' and data[col][i] == 'unknown':
            data[col][i] = 'single'
        # U stupcu 'education' ukoliko je nepoznat podatak pretpostavljamo da je osoba nije pohađala školu
        if col == 'education' and data[col][i] == 'unknown':
            data[col][i] = 'illiterate'
        # U stupcima 'default', 'housing', 'loan', ukoliko je podatak nepoznat postavljamo None kako bi ih uklonili
        if col in ["default", "housing", "loan"] and data[col][i] == 'unknown':
            data[col][i] = None
        # U stupcu 'duration', ukoliko je trajanje poziva manje od 35 s smatramo nepotrebnim
        if col == 'duration' and data[col][i] < 35:
            data[col][i] = None
        # U stupcu 'pdays' broj 999 znači da osoba nije kontaktirana i te retke ćemo označiti kao None
        if col == 'pdays' and data[col][i] == 999:
            data[col][i] = None
        # U stupcu 'poutcome', ukoliko ne postoje podatci o uspješnosti, brišemo retke
        if col == 'poutcome' and data[col][i] == 'nonexistent':
            data[col][i] = None


print(data.isna().sum())
# Na temelju gornjeg ispisa vidimo status podataka sa None
# Stupci 'pdays', 'poutcome' i 'previous' nam ne predstavljaju ništa
# značajno, te ćemo iste ukloniti potpuno
data.drop(['pdays', 'poutcome', 'previous'], axis=1, inplace=True)

# Sada ćemo ukloniti i retke koji imaju None vrijednosti u bilo kojem stupcu
# dropna je default any
data.dropna(axis=0, inplace=True)

print('Nakon uklanjanja None vrijednosti, status podataka je: ', end = '')
print(data.shape)

# Sada ćemo provjeriti koreliranost podataka
correlated_data = data.corr()
plt.figure(figsize= (12,8))
sns.heatmap(correlated_data, cmap="Greens", annot=True)
plt.show()

#Vidimo da su 'emp.var.rate', 'nr.employed' i 'euribor3m' korelirani, te ćemo njih izbaciti

data.drop(['emp.var.rate', 'euribor3m', 'nr.employed'], axis=1, inplace=True)

# Sada ćemo pretvoriti kategorijske i binarne podatke u numeričke

# Binarni podatci
binary_clms = ['default', 'housing', 'subscribed', 'loan']
for clm in binary_clms:
    list = []
    for txt in data[clm].values:
        if txt == 'yes':
            list.append(1)
        else:
            list.append(0)
    data[clm] = list

# Kategorijski podatci

job_types = {'admin.':1,'blue-collar':2,'entrepreneur':3,'housemaid':4,'management':5,'retired':6,'self-employed':7,
             'services':8,'student':9,'technician':10,'unemployed':11}
marital_status = {'divorced':1,'married':2,'single':3}
edu = {'basic.4y':1,'basic.6y':2,'basic.9y':3,'high.school':4,'illiterate':5,'professional.course':6,'university.degree':7}
contact_type = {'cellular':1,'telephone':2}
months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
days = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5}

#Pretvaranje kategorijskih podataka u numeričke pomoću pomoćne liste list
list = []
for txt in data['job'].values:
    list.append(job_types[txt])
data['job'] = list

list = []
for txt in data['marital'].values:
    list.append(marital_status[txt])
data['marital'] = list

list = []
for txt in data['education'].values:
    list.append(edu[txt])
data['education'] = list

list = []
for txt in data['contact'].values:
    list.append(contact_type[txt])
data['contact'] = list

list = []
for txt in data['month'].values:
    list.append(months[txt])
data['month'] = list

list = []
for txt in data['day_of_week'].values:
    list.append(days[txt])
data['day_of_week'] = list
cpy = []

print(data)

data.to_csv('arrangedData.csv', index = False)