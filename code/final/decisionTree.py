#ML Assignment
# Decision Tree

import numpy as np
import math
import copy

class DecisionTree:

	minNodes = 9 #size of set below which splitting stops
	numClasses = 9
	attribute_name_map = []

	class TreeNode:
		'''one node of the decision tree'''

		def __init__(self,splitted_indices,X,Y):

			self.splitted_indices = [] #indices which have already bin split along in ancestor nodes
			self.children = [] #each child would also be a tree node
			self.child_index = {} #dictionary mapping attribute value to index in children[]
			self.split_index = 0 #index of attribute alond which to split the data at this node
			self.leaf_node = False #true if current node is a leaf node
			self.prediction = -1 #if current node is leaf node, what is the predicted grade
			self.prediction_prob = 0 #probability of the prediction

			self.splitted_indices = copy.deepcopy(splitted_indices)
			if len(Y) <= DecisionTree.minNodes or (0 not in splitted_indices):
				if len(Y) <= DecisionTree.minNodes:
					print("Number of elements in set = ",len(Y),"is less than minNodes = ",DecisionTree.minNodes)
				else:
					print("Splitting has been done along all attributes")
				print("Stopping further split")
				self.leaf_node = True
				s = len(Y)
				counts = {10:0,9:0,8:0,7:0,6:0,5:0,4:0,2:0,0:0}	
				for y in Y:
					counts[y] = counts[y]+1
				for k in counts.keys():
					if self.prediction_prob < counts[k]/s:
						self.prediction_prob = counts[k]/s
						self.prediction = k
				print("predicted grade point for this node is",self.prediction,"with probability = ",self.prediction_prob)
				return

			cur_ent = DecisionTree.TreeNode.entropy(Y)
			print("current entropy: ",cur_ent)
			print("splitted indices",splitted_indices)
			best_gain_ratio =- 10000
			for i in range(len(splitted_indices)):
				if splitted_indices[i] == 1:
					continue
				current_gain_ratio = DecisionTree.TreeNode.gainRatio(i,cur_ent,X,Y)
				print("gain ratio when split along",DecisionTree.attribute_name_map[i]," = ", current_gain_ratio)
				if current_gain_ratio > best_gain_ratio:
					best_gain_ratio = current_gain_ratio
					self.split_index = i
			si = self.split_index
			print("\nBest split index is:",si,"attribute Name:",DecisionTree.attribute_name_map[si])
			self.splitted_indices[si] = 1

			imap = {}
			Xsub = []
			Ysub = []
			i = 0
			for j in range(len(X)):
				if X[j][si] not in imap.keys():
					imap[X[j][si]] = i
					Xsub.append([])
					Ysub.append([])
					i = i+1
				Xsub[imap[X[j][si]]].append(X[j])
				Ysub[imap[X[j][si]]].append(Y[j])

			self.child_index = imap

			print()
			print("====================================================================")
			print("Splitting along",DecisionTree.attribute_name_map[si])
			print("Creating",len(Ysub),"chilid nodes")

			for i in range(len(Xsub)):
				newChild = DecisionTree.TreeNode(self.splitted_indices,Xsub[i],Ysub[i])
				self.children.append(newChild)
				print()

			#default prediction incase input can't be split along this attribute.
			s = len(Y)
			counts = {10:0,9:0,8:0,7:0,6:0,5:0,4:0,2:0,0:0}	
			for y in Y:
				counts[y] = counts[y]+1
			for k in counts.keys():
				if self.prediction_prob < counts[k]/s:
					self.prediction_prob = counts[k]/s
					self.prediction = k
			return	


		def entropy(Y):
			s = len(Y)
			counts = {10:0,9:0,8:0,7:0,6:0,5:0,4:0,2:0,0:0}
			for y in Y:
				counts[y] = counts[y]+1
			e = 0
			for v in counts.values():
				if v != 0:
					e = e-(v/s)*math.log(v/s)
			return e

		def sizeEntropy(Ysub):
			s = 0;
			total = 0
			for sub in Ysub:
				total = total+len(sub)
			for sub in Ysub:
				if len(sub) != 0:
					s = s-(len(sub)/total)*math.log(len(sub)/total)
			return s

		def gainRatio(si,cur_ent, X,Y):
			imap = {}
			Ysub = []
			i = 0
			for j in range(len(X)):
				if X[j][si] not in imap.keys():
					imap[X[j][si]] = i
					Ysub.append([])
					i = i+1
				Ysub[imap[X[j][si]]].append(Y[j])
			S = DecisionTree.TreeNode.sizeEntropy(Ysub)
			E = 0
			total = 0
			for sub in Ysub:
				total = total+len(sub)
			for sub in Ysub:
				E = E+(len(sub)/total)*DecisionTree.TreeNode.entropy(sub)
			print("average Entropy when split along",DecisionTree.attribute_name_map[si],' = ',E)
			if S == 0: #only one value for this attribute so "split" along this is point less, so set gain to zero
				S = 100000000
			return (cur_ent-E)/S



	def __init__(self,X_train,Y_train,minNodes):
		'''assigns training data to class attributes'''

		self.X_train = []
		self.Y_train = []
		self.treeRoot =  None #root of decision tree
		self.attribute_name_map = []
		self.X_train,self.Y_train = DecisionTree.treeFilter(X_train,Y_train)
		DecisionTree.minNodes = minNodes
		DecisionTree.attribute_name_map.append("Year")
		DecisionTree.attribute_name_map.append("Gender")
		DecisionTree.attribute_name_map.append("Discretised midsem Score")
		DecisionTree.attribute_name_map.append("Midsem Grade")
		DecisionTree.attribute_name_map.append("Answer Sheet Collection")

	def train(self):
		'''Trains the model'''
		self.treeRoot = self.TreeNode([0,0,0,0,0],self.X_train,self.Y_train)

	def test(self,X_test,Y_test):
		'''returns Y_predicted and accuracy'''
		X_test,Y_test = DecisionTree.treeFilter(X_test,Y_test)
		Y_predicted = []
		Y_midsem = []
		for x in X_test:
			Y_predicted.append(self.predict(x))
			Y_midsem.append(x[-2])
		print("midsem error = ",DecisionTree.error(Y_midsem,Y_test))
		return Y_predicted,DecisionTree.error(Y_predicted,Y_test)

	def error(yp,yt):
		e = 0;
		for i in range(len(yp)):
			e = e+abs(yp[i]-yt[i])
		return e/len(yp)

	def predict(self,x):
		node = self.treeRoot
		# print(len(node.children))
		while node.leaf_node == False:
			if x[node.split_index] not in node.child_index.keys():
				return node.prediction
			node = node.children[node.child_index[x[node.split_index]]]
		return node.prediction

	def treeFilter(X_train,Y_train):
		'''process X_train to remove CGPA column 
		cause of too many missing values. Also discretises midsem marks'''
		new_X = []
		for x in X_train:
			gender = 0
			if x[1] == 'M':
				gender = 0
			else:
				gender = 1
			new_X.append([int(x[0]),gender,int(float(x[3])//5),int(x[4]),int(x[5])])

		new_Y = []
		for y in Y_train:
			new_Y.append(int(y))

		new_X = np.array(new_X)
		new_Y = np.array(new_Y)

		return new_X,new_Y

