I created following functions to implement adaBoost:
adaTrain - main function which returns H
adaPredict - It calculated accuracy on Test data
addTrainSingle - It helps to calculate H for perceptron and Decision stump, take care of all Et,Alphat and Dt+1 logic
DecisionStump - It is calculating best weak classifier among all option , returns ht and Ypred
Pla - It is calculating best weak classifier among all option through pocket algorithm , returns ht and Ypred

Assumption :
Replaced ? with random.choice[-1,1]
Replace "Democrates" with "1"
Replace "Republican" with "-1"
Replace "Yes" with "1"
Replace "No" with "-1"
Restricted pocket with 100 iteration 
Restricted adaboost for 15 times