__author__ = 'Krunal'
##[X, Y] = generateData(N)

import numpy as np
import sys
from numpy.linalg import inv
import csv

def generateData(N):
    X=2*np.random.random_sample((N,2)) - 1
    l0=2*np.random.random_sample((1,2))-1
    l1=2*np.random.random_sample((1,2))-1
    count = 1
    Y=np.array([])
    for i in range(len(X)):
        if (X[i][0] - l0[0][0]) * (l1[0][1] - l0[0][1]) - (X[i][1] - l0[0][1]) * (l1[0][0] - l0[0][0])>0:
            Y= np.append(Y,1)
        else:
            Y= np.append(Y,-1)
        count=count+1
    return X,Y

def pla(X,Y,w0):
    iteration = 0
    w = w0
    # Introduce an artificial coordinate x0= 1
    X0 = np.ones((len(X),1))
    X=np.concatenate((X0,X),axis=1)
    # Implementation of Sign(WtX)
    IsConverging = True
    while(IsConverging):
        Ypred = np.sign(X.dot(w.T))
        # Pick a misclassified point
        if np.all(Y == Ypred):
            IsConverging = False
        else: # Repeat process till model converged, increase count of iteration
            misclassfied = np.array([]) # Initializd array to store misclassified values
            for i in range(len(X)):
                if Ypred[i]!=Y[i]:
                    misclassfied=np.append(misclassfied,i) # assigned index of mis classified values
            temp=np.random.randint(0,len(misclassfied),size=1) # random selection
            temp1=int(misclassfied[temp[0]]) # finding missclassifed point
            w=w+Y[temp1]*X[temp1] # updating weight due to mis classified point
            iteration = iteration + 1
    #print "Yprediction :- "
    #print Ypred
    return w,iteration

def pseudoinverse(X, Y):
    # w = XdaggerY , Xdagger=(XT.X)-1.XT
    # Introduce an artificial coordinate x0= 1
    X0 = np.ones((len(X),1))
    X=np.concatenate((X0,X),axis=1)
    Xinv = inv(X.T.dot(X))
    Xdagger = Xinv.dot(X.T)
    w=Xdagger.dot(Y)
    return w

def main(args):
    N = [10,50]
    csvout = csv.writer(open("C:\Users\Krunal\Documents\DSBA\Spring 2016\ML\Hw2\mydata.csv", "wb"))
    csvout.writerow(("i","N","Iter w/0 R", "Iter w R"))
    for n in N:
        for i in range(1,101):
            [X,Y]=generateData(n)
            #Initialize W0 weight vector
            #print "##### Experiment 1 #####"
            w0=np.zeros(1+X.shape[1])
            [W,Iteration]=pla(X,Y,w0)
            #print "Leanred Weight Vector without Regression"
            #print W
            #print "iterations without Regression"
            #print Iteration
            wl0=pseudoinverse(X, Y)
            #print "##### Experiment 2 #####"
            #print "Weight from pseudoInverse algorithm"
            #print wl0
            [W1,Iteration1]=pla(X,Y,wl0)
            csvout.writerow((i,n,Iteration,Iteration1))
            #print "Leanred Weight Vector with Regression"
            #print W1
            #print "iterations with Regression"
            #print Iteration1

if __name__ == '__main__':
    main(sys.argv)
