__author__ = 'Krunal'

import numpy as np
import sys
from numpy.linalg import inv
import csv
import pandas as pd
from sklearn.cross_validation import train_test_split
import math
import random

#model = adaTrain(XTrain, YTrain, version) which takes as input:
#XTrain, the training examples, a (N, D) numpy ndarray where N is the number of training examples and D is the dimensionality
#YTrain, a 1-D array of labels
#version, a string which can be 'stump' 'perceptron' or 'both'
#This method returns a model object containing the parameters of the trained model.
def DecisionStump(X,Y,Dt):
    #calculate entropy and information gain to get first attribute
    #calculate YPred out of first attribute
    y,rcount= np.unique(Y,return_counts=True)
    YCount= zip(y,rcount)
    P0,P1= 1.0*YCount[0][1]/len(Y),1.0*YCount[1][1]/len(Y)
    EntropyofY = -P0*np.log2(P0)-P1*np.log2(P1)
    #print YCount,YCount[0][1],YCount[1][1],len(Y),P0,P1,EntropyofY,X.shape[1]
    XT = X.T
    InfromationGain = np.zeros(16)
    for i in range(XT.shape[0]):
            x,rcount= np.unique(XT[i],return_counts=True)
            XCount= zip(x,rcount)
            P0,P1= 1.0*XCount[0][1]/len(X),1.0*XCount[1][1]/len(X)
            tab = np.histogram2d(Y,XT[i],bins=2,weights=Dt)
            print tab
            #print XCount[0][1],tab[0][0][0],tab[0][0][1],XCount[1][1],tab[0][1][0],tab[0][1][1]
            P00,P01,P10,P11=tab[0][0][0]/XCount[0][1],tab[0][0][1]/XCount[0][1],tab[0][1][0]/XCount[1][1],tab[0][1][1]/XCount[1][1]
                #np.array([np.bincount(x[Y==y]) for y in np.unique(Y)])
            #print tab,P0,P1,P00,P01,P10,P11
            if P00==0: temp1=0
            else: temp1=P00 * np.log2(P00)
            if P01==0: temp2=0
            else: temp2=-P01 * np.log2(P01)
            if P10==0: temp3=0
            else: temp3=-P10 * np.log2(P10)
            if P11==0: temp4=0
            else: temp4=-P11 * np.log2(P11)
            EntropyofX = (-temp1 - temp2)*P0 + (-temp3 - temp4)*P1
            InfromationGain[i] = EntropyofY - EntropyofX
    print InfromationGain
    print 'key attribute'
    print np.argmax(InfromationGain)
    temp= np.histogram2d(Y,XT[np.argmax(InfromationGain)],bins=2,weights=Dt)
    print 'temp'
    tempT = temp[0].T
    print tempT
    print tempT[0][0],tempT[0][1],tempT[1][0],tempT[1][1]
    Ypred = np.zeros(len(Y))
    #print XT[np.argmax(InfromationGain)][0],len(Y), len(XT[np.argmax(InfromationGain)])
    temp1=np.argmax(tempT)
    print temp1
    if tempT[0][0]>tempT[0][1]: t1=-1
    else:  t1=1
    if tempT[1][0]>tempT[1][1]: t2=-1
    else:  t2=1
    ht=[np.argmax(InfromationGain),t1,t2]
    for i in range(len(X)):
        if XT[np.argmax(InfromationGain)][i]==ht[1]: Ypred[i]=ht[2]
        else:  Ypred[i]=-ht[2]
    #error=modelerror(Y,Ypred)
    #print "DS Train"
    #print error
    return ht,Ypred

def DecisionStump2(X,Y,Dt):
    #calculate entropy and information gain to get first attribute
    #calculate YPred out of first attribute
    XT = X.T
    Ypred = np.zeros(len(Y))
    ht= []
    for i in range(len(XT)):
        temp= np.histogram2d(Y,XT[i],bins=2,weights=Dt)
        #print temp
        #temp1=np.argmax(temp[0])
        tempT = temp[0].T
        temp1=np.argmax(tempT)
        #print temp1
        if tempT[0][0]>tempT[0][1]: t1=-1
        else:  t1=1
        if tempT[1][0]>tempT[1][1]: t2=-1
        else:  t2=1
        ht.append([i,t1,t2])

    error = np.empty(len(XT))
    for j in range(len(XT)):
        for i in range(len(X)):
            if XT[j][i]==ht[j][0]: Ypred[i]=ht[j][1]
            else:  Ypred[i]=-ht[j][1]
        error[j]=modelerror(Y,Ypred)
    bestIndex= np.argmin(error)
    for i in range(len(X)):
        if XT[bestIndex][i]==ht[bestIndex][1]: Ypred[i]=ht[bestIndex][2]
        else:  Ypred[i]=-ht[bestIndex][2]
    return ht[bestIndex],Ypred


def pseudoinverse(X, Y):
    # w = XdaggerY , Xdagger=(XT.X)-1.XT
    # Introduce an artificial coordinate x0= 1
    X0 = np.ones((len(X),1))
    X=np.concatenate((X0,X),axis=1)
    Xinv = inv(X.T.dot(X))
    Xdagger = Xinv.dot(X.T)
    w=Xdagger.dot(Y)
    return w

def pla(X,Y,w,Dt):
    iteration = 0
    # Introduce an artificial coordinate x0= 1
    X0 = np.ones((len(X),1))
    X=np.concatenate((X0,X),axis=1)
    # Implementation of Sign(WtX)
    IsConverging = True
    preverror = 0
    while(IsConverging and iteration <100):
        Ypred = np.sign(X.dot(w.T))
        # Pick a misclassified point
        if np.all(Y == Ypred):
            IsConverging = False
        else: # Repeat process till model converged, increase count of iteration
            misclassfied = np.array([]) # Initializd array to store misclassified values
            prevDt=0
            temp1=0
            for i in range(len(X)):
                if Ypred[i]!=Y[i]:
                    if(Dt[i] > prevDt):
                        prevDt = Dt[i]
                        temp1=i
                    elif(Dt[i] == prevDt):
                        temp1=i
            w=w+Y[temp1]*X[temp1] # updating weight due to mis classified point
            curerror = modelerror(Y,Ypred)
            if iteration == 0 : curerror=preverror
            if curerror >= preverror:
                wfinal=w
        iteration = iteration + 1
    #modelerror(Y,Ypred)
    return wfinal,Ypred

def modelerror(YActual, YPred): # check accuracy of model
        fp = 0
        for x in range(len(YActual)):
            if YActual[x] != YPred[x]:
                fp += 1 # if its is matching then increasing count
        total = float(len(YActual))
        #print '% of misclassified'
        #print (fp/total) # actual / total divide
        return (fp/total)*100

def modelerrorfromH(H,X,Y,version):
    signofH = np.zeros(len(Y))
    TYpred = np.zeros(len(Y))
    Ypred = np.zeros(len(Y))
    if version == 'Perceptron':
        X0 = np.ones((len(X),1))
        X=np.concatenate((X0,X),axis=1)
    #print H
    for h in H:
            #print 'Hypothesis'
            #print h,h[0][0],h[0][1],h[0][2],h[1],h[2]
            if h[2] == 'Stump':
                XT=X.T
                for i in range(len(Y)):
                    if XT[h[0][0]][i]==-1: TYpred[i]=h[0][1]
                    else: TYpred[i]=h[0][2]
                signofH = signofH + h[1]*TYpred
                Ypred=np.sign(signofH)
                #print Ypred
            elif h[2] == 'Perceptron':
                #print X.shape,h[0],h[1],h[2]
                TYpred=np.sign(X.dot(h[0].T))
                signofH = signofH + h[1]*TYpred
                Ypred=np.sign(signofH)
    error=modelerror(Y,Ypred)
    return error

def adaTrain(XTrain,YTrain,version):
    if(version == "Stump"):
            #print loop
            H= adaTrainsingle(XTrain,YTrain,version)
    elif(version == "Perceptron"):
            H= adaTrainsingle(XTrain,YTrain,version)
    elif(version == "Both"):
            P1=np.empty(len(XTrain))
            P2=np.empty(len(XTrain))
            H1= adaTrainsingle(XTrain,YTrain,'Stump')
            Ypred1 = adaPredict(H1,XTrain)
            H2= adaTrainsingle(XTrain,YTrain,'Perceptron')
            Ypred2 = adaPredict(H2,XTrain)
            for i in range(len(XTrain)):
                P1[i]= 1.0/ (1 + math.exp(-1.0*Ypred1[i]))
                P2[i]= 1.0/ (1 + math.exp(-1.0*Ypred2[i]))
            H=H2
            #H = 1 - (1-P1)*(1-P2)
            #print H
    return H

def adaTrainsingle(XTrain,YTrain,version):
    XATrain = XTrain[:len(XTrain)*0.75]
    YATrain = YTrain[:len(YTrain)*0.75]
    XValidation = XTrain[len(XTrain)*0.75:]
    YValidation = YTrain[len(YTrain)*0.75:]
    Dt = np.empty(len(XATrain))
    Dt.fill(1.0 / len(XATrain))
    Dt1 = np.empty(len(XATrain))
    loop = 0
    H = []
    prevmodelerror=0
    IsStop=True
    currentmodelerror= np.empty(15)
    while(loop < 15): # loop==0 or IsStop validation condition to stop ensemble
        if(version == "Stump"):
            #print loop
            #ht,Ypred= DecisionStump(XATrain,YATrain,Dt)
            ht,Ypred=DecisionStump2(XATrain,YATrain,Dt)
            #print ht
        elif(version == "Perceptron"):
            w=pseudoinverse(XATrain, YATrain)
            ht,Ypred = pla(XATrain,YATrain,w,Dt)

        #print w
        Et=0.001
        if Et != 0:
            for i in range(len(YATrain)):
                if YATrain[i]!=Ypred[i]: Et= Et + Dt[i]

        Alphat= 0.5*math.log((1-Et)/Et,2)
        #print Et,Alphat,len(Dt)
        Zt=0.0
        temp = 0.0
        Dt1temp = np.zeros(len(YATrain))
        for i in range(len(Dt)):
            Dt1temp[i] = Dt[i]*math.exp(-Alphat*YATrain[i]*Ypred[i])
            Zt=Zt + Dt1temp[i]
        for i in range(len(Dt1temp)):
            Dt1[i]= Dt1temp[i]/Zt
            temp= temp + Dt1[i]
        Dt=Dt1
        #print ht,Et,Alphat

        H.append((ht,Alphat,version))

        currentmodelerror[loop]=modelerrorfromH(H,XValidation,YValidation,version)
        loop = loop + 1
        #print 'validation error'
        #print currentmodelerror
        #if loop==0: prevmodelerror=currentmodelerror
        #if currentmodelerror < prevmodelerror: prevmodelerror=currentmodelerror
        #else: IsStop=False
    #print np.argmin(currentmodelerror)
    for j in range(14-(np.argmin(currentmodelerror))):
        H.pop()
    return H

def modelaccuracy(testY, predictiontestY): # check accuracy of model
        tp = 0
        for x in range(len(testY)):
            if testY[x] == predictiontestY[x]:
                tp += 1 # if its is matching then increasing count
        total = float(len(testY))
        return (tp/total) * 100.0 # actual / total divide

#the model object returned by adaTrain
#XTest, the testing examples, a (N, D) numpy ndarray where N is the number of testing examples and D is the dimensionality
#This method returns a 1-D array of predicted labels corresponding to the provided test examples.
def adaPredict(model, XTest):
    signofH = np.zeros(len(XTest))
    TYpred = np.zeros(len(XTest))
    Ypred = np.zeros(len(XTest))
    X0 = np.ones((len(XTest),1))
    X=np.concatenate((X0,XTest),axis=1)
    for h in model:
            #print 'Hypothesis'
            #print h,h[0][0],h[0][1],h[0][2],h[1],h[0],h[2]
            if h[2] == 'Stump':
                XT=XTest.T
                for i in range(len(XTest)):
                    if XT[h[0][0]][i]==h[0][1]: TYpred[i]=h[0][2]
                    else: TYpred[i]=-h[0][2]
                signofH = signofH + h[1]*TYpred
                Ypred=np.sign(signofH)
                Ypred=TYpred
            elif h[2] == 'Perceptron':
                #print X.shape,h[0],h[1],h[2]
                TYpred=np.sign(X.dot(h[0].T))
                signofH = signofH + h[1]*TYpred
                Ypred=np.sign(signofH)
                #print TYpred
                #np.append(Ypred,TYpred)
    return Ypred

def main(args):
    df = pd.read_csv('C:\Users\Krunal\Documents\DSBA\Spring 2016\ML\house-votes-84.data.txt', header=None)
    #randomindex = np.random.rand(len(df)) <= 0.75
    #train = df.sample(frac=0.75, random_state=320)
    df=df.replace("republican",-1)
    df=df.replace("democrat",1)
    df=df.replace("y",1)
    df=df.replace("n",-1)
    df=df.replace("?",random.choice([1,-1]))
    df1 = np.array(df)
    testerror = np.empty(10)
    testacc = np.empty(10)
    version = 'Stump'
    print version
    for i in range(10):
        np.random.shuffle(df1)
        XTrain = df1[:len(df1)*0.75,1:]
        YTrain = df1[:len(df1)*0.75,0]
        XTest = df1[len(df1)*0.75:,1:]
        YTest = df1[len(df1)*0.75:,0]
        H=adaTrain(XTrain,YTrain,version)
        Ypred=adaPredict(H,XTest)
        #print Ypred
        testerror[i]=modelerror(YTest,Ypred)
        testacc[i]=modelaccuracy(YTest,Ypred)
    print '% of misclassified of Test'
    print testerror
    print 'Test accuracy'
    print testacc
    print 'Best'
    print np.max(testacc)
    print 'Worst'
    print np.min(testacc)
    print 'avg'
    print np.average(testacc)



if __name__ == '__main__':
    main(sys.argv)
