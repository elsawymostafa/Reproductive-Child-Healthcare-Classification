# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 14:01:54 2022

@author: elsaw
"""
#import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tabulate import tabulate
from IPython.display import display

def tp(Ytest,Yprdect):
    count =0
    a=0
    for i in  Yprdect:
        if i!=3.0 and Ytest[a]!=3.0:
            count=count+1
        a=a+1
    return count

def fn(Ytest,Yprdect):
    count =0
    a=0
    for i in  Yprdect:
        if i==3.0 and Ytest[a]!=3.0:
            count=count+1
        a=a+1
    return count

def tn(Ytest,Yprdect):
    count =0
    a=0
    for i in  Yprdect:
        if i==3.0 and Ytest[a]==3.0:
            count=count+1
        a=a+1
    return count

def fp(Ytest,Yprdect):
    count =0
    a=0
    for i in  Yprdect:
        if i!=3.0 and Ytest[a]==3.0:
            count=count+1
        a=a+1
    return count

def recall (Ytest,Yprdect):
    return tp(Ytest,Yprdect)/(tp(Ytest,Yprdect)+fn(Ytest,Yprdect))

def acuercy (Ytest,Yprdect):
    return (tp(Ytest,Yprdect)+tn(Ytest,Yprdect))/Yprdect.shape[0]

def precision (Ytest,Yprdect):
    return tp(Ytest,Yprdect)/(tp(Ytest,Yprdect)+fp(Ytest,Yprdect))

def f1Scor (Ytest,Yprdect):
    return 2*((precision (Ytest,Yprdect)*recall (Ytest,Yprdect))/(recall (Ytest,Yprdect)+precision (Ytest,Yprdect)))

df = pd.read_csv("train_dataset.csv")
X = df[['baseline value','accelerations','fetal_movement','uterine_contractions','light_decelerations','severe_decelerations','prolongued_decelerations','abnormal_short_term_variability','mean_value_of_short_term_variability','percentage_of_time_with_abnormal_long_term_variability']]
Y=df[['fetal_health']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)        
acc={}
rec={}
per={}
f1={}
er={}
accl=[]
clfn = MLPClassifier()
clfn.fit(X_train, Y_train)
p=clfn.predict(X_test)
list=Y_train.to_numpy()
print ("             Neurel Network")
acc.update({"NN":acuercy(list,p)})
rec.update({"NN":recall (list,p)})
per.update({"NN":precision(list,p)})
f1.update({"NN":f1Scor(list,p)})
er.update({"NN":1-acuercy(list,p)})
print ("Acuercy=    ",acc["NN"])
print ("Recall=     ",rec["NN"])
print ("Percision=  ",per["NN"])
print ("F1 Score=   ",f1["NN"])
print ("Error Rate= ",1-acc["NN"])

clflgr = LogisticRegression()
clflgr.fit(X_train, Y_train)
p=clflgr.predict(X_test)
list=Y_train.to_numpy()
print ("             Logistic Regression")
acc.update({"LR":acuercy(list,p)})
rec.update({"LR":recall (list,p)})
per.update({"LR":precision(list,p)})
f1.update({"LR":f1Scor(list,p)})
er.update({"LR":1-acuercy(list,p)})
print ("Acuercy=    ",acc["LR"])
print ("Recall=     ",rec["LR"])
print ("Percision=  ",per["LR"])
print ("F1 Score=   ",f1["LR"])
print ("Error Rate= ",1-acc["LR"])

clfdt = DecisionTreeClassifier()
clfdt.fit(X_train, Y_train)
p=clfdt.predict(X_test)
list=Y_train.to_numpy()
print ("             Decision Trees")
acc.update({"DT":acuercy(list,p)})
rec.update({"DT":recall (list,p)})
per.update({"DT":precision(list,p)})
f1.update({"DT":f1Scor(list,p)})
er.update({"DT":1-acuercy(list,p)})
print ("Acuercy=    ",acc["DT"])
print ("Recall=     ",rec["DT"])
print ("Percision=  ",per["DT"])
print ("F1 Score=   ",f1["DT"])
print ("Error Rate= ",1-acc["DT"])

clfsv = SVC()
clfsv.fit(X_train, Y_train)
p=clfsv.predict(X_test)
list=Y_train.to_numpy()
print ("             Support Vector Machines")
acc.update({"SVM":acuercy(list,p)})
rec.update({"SVM":recall (list,p)})
per.update({"SVM":precision(list,p)})
f1.update({"SVM":f1Scor(list,p)})
er.update({"SVM":1-acuercy(list,p)})
print ("Acuercy=    ",acc["SVM"])
print ("Recall=     ",rec["SVM"])
print ("Percision=  ",per["SVM"])
print ("F1 Score=   ",f1["SVM"])
print ("Error Rate= ",1-acc["SVM"])

clfxgb= GradientBoostingClassifier(n_estimators=100,learning_rate=1.0, max_depth=1,random_state=0)
clfxgb.fit(X_train, Y_train)
p=clfxgb.predict(X_test)
print ("             XGboost")
acc.update({"XGBOOST":acuercy(list,p)})
rec.update({"XGBOOST":recall (list,p)})
per.update({"XGBOOST":precision(list,p)})
f1.update({"XGBOOST":f1Scor(list,p)})
er.update({"XGBOOST":1-acuercy(list,p)})
print ("Acuercy=    ",acc["XGBOOST"])
print ("Recall=     ",rec["XGBOOST"])
print ("Percision=  ",per["XGBOOST"])
print ("F1 Score=   ",f1["XGBOOST"])
print ("Error Rate= ",1-acc["XGBOOST"])

figure(figsize=(8,15))
plt.bar(acc.keys(),acc.values(),width =0.3)
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel("Accuracy")
plt.savefig('acc.png')
plt.show()

figure(figsize=(8,15))
plt.bar(rec.keys(),rec.values(),width =0.3)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Recall")
plt.savefig('rec.png')
plt.show()

figure(figsize=(8,15))
plt.bar(per.keys(),per.values(),width =0.3)
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel("Percision")
plt.savefig('per.png')
plt.show()

figure(figsize=(8,15))
plt.bar(f1.keys(),f1.values(),width =0.3)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("F1 Score")
plt.savefig('f1.png')
plt.show()

figure(figsize=(8,15))
plt.bar(er.keys(),er.values(),width =0.3)
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel("Erorr Rate")
plt.savefig('er.png')
plt.show()
d={"Model":acc.keys(),"Accuracy":acc.values()}
df=pd.DataFrame(data=d)
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

d={"Model":rec.keys(),"Recall":rec.values()}
df=pd.DataFrame(data=d)
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

d={"Model":per.keys(),"Percision":per.values()}
df=pd.DataFrame(data=d)
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

d={"Model":f1.keys(),"F1 Score":f1.values()}
df=pd.DataFrame(data=d)
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

d={"Model":er.keys(),"Error Rate":er.values()}
df=pd.DataFrame(data=d)
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

d={"Model":acc.keys(),"Accuracy":acc.values(),"Recall":rec.values(),"Percision":per.values(),"F1 Score":f1.values(),"Error Rate":er.values()}
df=pd.DataFrame(data=d)
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))