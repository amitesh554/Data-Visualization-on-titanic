import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('train.csv')


#Correctness of data
graph=sns.FacetGrid(df,col="Survived")
df.loc[df["Fare"]>400,"Fare"]=df["Fare"].median()
graph.map(plt.hist,"Fare",bins=20)
plt.show()

graph_age=sns.FacetGrid(df,col="Survived",palette="Red")
df.loc[df["Age"]>70,"Age"]=df["Age"].median()
graph_age.map(plt.hist,"Age",bins=20)
plt.show()

graph_pclass=sns.FacetGrid(df,col="Survived")
graph_pclass.map(plt.hist,"Pclass",bins=20)
plt.show()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Completeness of data

for column in df:
    print(column,":",df[column].isnull().sum())
    
df["Age"].fillna(df["Age"].median(),inplace=True)

print(df["Embarked"].value_counts())
df["Embarked"].fillna("S",inplace=True)

del df["Cabin"]

#////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Correctness of Data

df.sample(50)

def get_title(name):
    if "." in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "No title in name"

title=set([x for x in df.Name.map(lambda x:get_title(x))])

print(title)

def shorten_title(x):
    title=x["Title"]
    if title in ['Capt','Col','Major']:
        return "Officer"
    elif title in ['Jonkheer','Don','the Countess','Dr','Lady','Sir']:
        return "Royalty"
    elif title in 'Mme':
        return 'Mrs'
    elif title in ['Mlle','Ms']:
        return "Miss"
    else:
        return title
        

df['Title']=df['Name'].map(lambda x:get_title(x))

df['Title']=df.apply(shorten_title,axis=1)

print(df.Title.value_counts())

df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.sample(20)


#Converting
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

df.Sex.replace(('male','female'),(0,1),inplace=True)
df.Embarked.replace(('S','C','Q'),(0,1,2),inplace=True)
df.Title.value_counts()

df.Title.replace(('Mr','Miss','Mrs','Master','Royalty','Rev','Officer'),(0,1,2,3,4,5,6),inplace=True)

df.sample(20)

