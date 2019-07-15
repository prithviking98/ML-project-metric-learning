import preprocess as pp
import decisionTree as dt
import nbc
import random
import time

random.seed()

file = 'data.csv'

rows = pp.readFile(file = file)

X_train,Y_train,X_test,Y_test = pp.splitData(rows, splitFraction = 0.80)

st1 = time.time()
tree = dt.DecisionTree(X_train,Y_train,9)
tree.train()
p1 = time.time()
Y_pred,error= tree.test(X_test,Y_test)
e1 = time.time()
print("error in decision tree testing = ",error)
# for i in range(len(Y_test)):
# 	print(Y_pred[i],Y_test[i])

st2 = time.time()
nbc = nbc.NBC()
nbc.train(X_train,Y_train)
p2 = time.time()
error, Y_predict = nbc.test(X_test,Y_test)
e2 = time.time()
# print(Y_predict,Y_test)
print("error in nbc testing = ",error)

#run times
print("%.6f %.6f %0.6f"%(p1-st1,e1-p1,e1-st1))
print("%.6f %.6f %0.6f"%(p2-st2,e2-p2,e2-st2))