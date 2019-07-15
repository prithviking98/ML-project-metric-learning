import numpy as np
import collections
import copy

class NBC():
	def __init__(self, whichProbability = [(0,-1),(0,-1),(0.0,0.0),(0.0,0.0),(0,-1),(0,-1),(0,-1)], newX_train = None, newY_train = None):
		self.X_train = list(newX_train) if newX_train != None else []
		self.Y_train = list(newY_train) if newY_train != None else []
		self.X = [{}]
		self.Y = {}
		self.X_probability = []
		self.Y_probability = []
		self.whichProbability = whichProbability

	def addData(self, newX_train, newY_train):
		self.X_train.extend(newX_train)
		self.Y_train.extend(newY_train)

		return

	def train(self, newX_train = None, newY_train = None):
		self.X_train.extend(newX_train)
		self.Y_train.extend(newY_train)
		X_train = np.array(self.X_train)
		Y_train = np.array(self.Y_train)
		
		n = X_train.shape[0]
		m = X_train.shape[1]
		
		self.X = [set(X_train[:,j]) if self.whichProbability[j][1] == -1 else set() for j in range(m)]
		self.Y = set(Y_train)
		
		self.X_probability = []
		Y_count = collections.Counter(Y_train)
		nullMean = np.mean(np.array(Y_train, dtype = float))
		nullStd = np.std(np.array(Y_train, dtype = float))
		for j in range(m):
			if self.whichProbability[j][1] == -1:
				newDict = {x: collections.Counter(self.Y) for x in self.X[j]}
				for i in range(n):
					newDict[X_train[i][j]][Y_train[i]] += 1

				for x in self.X[j]:
					for y in self.Y:
						newDict[x][y] = (newDict[x][y]-1)/Y_count[y]
						if newDict[x][y] == 0:
							newDict[x][y] = 0.001

				self.X_probability.append(newDict)
			else:
				classNums = {y: [] for y in self.Y}
				for i in range(n):
					classNums[Y_train[i]].append(X_train[i][j])
				
				newDict = {y: (0.0,0.0) for y in self.Y}
				for y in self.Y:
					classNums[y] = np.array(classNums[y], dtype = float)
					mean = np.mean(classNums[y])
					std = np.std(classNums[y])
					newDict[y] = (mean,std)

				self.X_probability.append(newDict)

		total = sum(Y_count.values())
		self.Y_probability = {y: y_count/total for y,y_count in Y_count.items()}

		return

	def getProbability(self, j, x, Ci):
		if self.whichProbability[j][1] == -1:
			return self.X_probability[j].get(x, collections.Counter())[Ci]
		else:
			mean = self.X_probability[j][Ci][0]
			std = self.X_probability[j][Ci][1]
			if mean == 0.0:
				return 1/(len(self.X_train)+1)
			elif std == 0.0:
				return 0.99 if x == mean else 1/(len(self.X_train)+1)
			return (1/(2*np.pi*(std**2))**0.5)*np.exp(-((float(x)-mean)**2)/(2*(std**2)))

	def predict(self, X):
		m = len(X)

		prob = [1.0 for i in range(len(self.Y_probability.items()))]
		index = 0
		Y_list = [Ci for Ci in self.Y]
		predictedClass = Y_list[0]
		for i in range(len(Y_list)):
			Ci = Y_list[i]
			prob[i] = self.Y_probability[Ci]
			for j in range(m):
				x = X[j]
				if x == None or x == '' or x == 0:
					continue
				prob[i] = prob[i] * self.getProbability(j, x, Ci)

			if prob[i] > prob[index]:
				index = i
				predictedClass = Ci

		return prob[index], predictedClass

	def test(self, X_test, Y_test):
		n = X_test.shape[0]
		m = X_test.shape[1]

		Y_predict = []
		for i in range(n):
			X = X_test[i]
			_, c = self.predict(X)
			Y_predict.append(int(c))

		Y_actual = []
		for y in Y_test:
			Y_actual.append(int(y))

		Y_predict = np.array(Y_predict)
		Y_actual = np.array(Y_actual)

		return sum(abs(Y_predict - Y_actual))/n, Y_predict