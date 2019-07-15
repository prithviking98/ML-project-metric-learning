# import sklearn.naive_bayes
import numpy as np
import csv
import math
import random
import copy


def readFile(file):
	csvfile = open(file, newline='')
	rows = np.array(list(csv.reader(csvfile, delimiter=',', quotechar='"')))
	newRows = []

	for r in rows:
		if r[6] == '': #if midsem not there, discard entry
			continue
		if r[4] == '': #if cg not given, map to zero for easier processing
			r[4] = '0'

		newRows.append([r[1],r[3],math.ceil(float(r[4])),r[5],grademap(r[6]),r[7],grademap(r[-3])])

	return newRows

def splitData(data, splitFraction = 0.8):
	'''splits data into training and testing by a factor of splitFraction'''
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

def grademap(grade):
	'''maps grade to gradepoint'''
	gradep=0
	if grade=='A':
		gradep=10
	elif grade=='A-':
		gradep=9
	elif grade=='B':
		gradep=8
	elif grade=='B-':
		gradep=7
	elif grade=='C':
		gradep=6
	elif grade=='D':
		gradep=5
	elif grade=='D-':
		gradep=4
	elif grade=='E':
		gradep=2
	elif grade=='NC':
		gradep=0

	return gradep
