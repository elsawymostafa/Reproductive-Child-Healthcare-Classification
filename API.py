# -*- coding: utf-8 -*-
"""
@author: elsaw
"""
from flask import *
import json ,time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

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

def acuercy (Ytest,Yprdect):
    return (tp(Ytest,Yprdect)+tn(Ytest,Yprdect))/Yprdect.shape[0]

df = pd.read_csv("train_dataset.csv")
X = df[['baseline value','accelerations','fetal_movement','uterine_contractions','light_decelerations','severe_decelerations','prolongued_decelerations','abnormal_short_term_variability','mean_value_of_short_term_variability','percentage_of_time_with_abnormal_long_term_variability']]
Y=df[['fetal_health']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)        
acc={}
clfn = MLPClassifier()
clfn.fit(X_train, Y_train)
p=clfn.predict(X_test)
list=Y_train.to_numpy()
acc.update({"NN":acuercy(list,p)})

clflgr = LogisticRegression()
clflgr.fit(X_train, Y_train)
p=clflgr.predict(X_test)
list=Y_train.to_numpy()
acc.update({"LR":acuercy(list,p)})

clfdt = DecisionTreeClassifier()
clfdt.fit(X_train, Y_train)
p=clfdt.predict(X_test)
list=Y_train.to_numpy()
acc.update({"DT":acuercy(list,p)})

clfsv = SVC()
clfsv.fit(X_train, Y_train)
p=clfsv.predict(X_test)
list=Y_train.to_numpy()
acc.update({"SVM":acuercy(list,p)})

clfxgb= GradientBoostingClassifier(n_estimators=100,learning_rate=1.0, max_depth=1,random_state=0)
clfxgb.fit(X_train, Y_train)
p=clfxgb.predict(X_test)
acc.update({"XGBOOST":acuercy(list,p)})

#print (clfn.predict([[133,0.003,0,0.004,0.004,0,0,30,1.5,0]]))

def prediction (data):
    result ={}
    result.update({"NN": [clfn.predict([data])[0],acc["NN"]]}) 
    result.update({"LR": [clflgr.predict([data])[0],acc["LR"]]}) 
    result.update({"DT": [clfn.predict([data])[0],acc["DT"]]}) 
    result.update({"SVM": [clfsv.predict([data])[0],acc["SVM"]]}) 
    result.update({"XGBOOST": [clfxgb.predict([data])[0],acc["XGBOOST"]]})
    return result
datac=[141,	0	,0.008	,0,	0,	0,	0	,75,	0.3	,49]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    josn_dump=json.dumps(prediction(datac))
    return josn_dump
@app.route('/get',methods=['GET'])
def get_request():
    list=[]
    use_input=request.data()
    for key in use_input:
    	list.append(use_input[key])
    josn_dump=json.dumps(prediction(list))
    return josn_dump

if __name__ == '__main__':
    app.run(port=80)#,debug=True)