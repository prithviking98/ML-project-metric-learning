# import sklearn.naive_bayes
import numpy as np
import csv
import math
import random
import copy

gradepoint = {'A': 10, 'A-': 9, 'B': 8, 'B-': 7, 'C': 6, 'C-': 5, 'D': 4, 'E': 2, 'NC': 0 }

def readFile(file):
	csvfile = open(file, newline='')
	rows = np.array(list(csv.reader(csvfile, delimiter=',', quotechar='"')))
	
	newRows = []
	for r in rows:
		if r[6] == '':
			continue
		if r[4] == '':
			r[4] = '0'

		newRows.append([r[1],
						r[3],
						math.ceil(float(r[4])),
						float(r[5]),
						gradepoint[r[6]],
						r[7],
						gradepoint[r[-3]]])

	return newRows

def splitData(data, splitFraction = 0.8):
	test = copy.deepcopy(data)
	random.shuffle(test)
	
	l1 = math.ceil(len(data)*splitFraction)
	train = np.array(test[:l1])
	test = np.array(test[l1:])
	
	X_train = train[:,:-1]
	Y_train = train[:,-1]
	X_test = test[:,:-1]
	Y_test = test[:,-1]

	return X_train,Y_train,X_test,Y_test
