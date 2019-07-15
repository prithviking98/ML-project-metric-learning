import preprocess as pp
import decisionTree as dt
import numpy as np
import random
random.seed()
file = 'data.csv'

rows = pp.readFile(file = file)

X_train,Y_train,X_test,Y_test = pp.splitData(rows)

tree = dt.DecisionTree(X_train,Y_train,9)
tree.train()
Y_pred,error= tree.test(X_test,Y_test)

print("error in decision tree testing =",error)
for i in range(len(Y_test)):
	print(Y_pred[i],Y_test[i])

print("vignesh's prediction")

Xmale=[['3','9.6','M','28.3','7','1']]
Xfemale=[['3','9.6','F','28.3','7','1']]
Y=[10]

yp,e=tree.test(Xmale,Y)
print("vignesh's grade",yp[0])
yp,e=tree.test(Xfemale,Y)
print("chhakka vignesh's grade",yp[0])