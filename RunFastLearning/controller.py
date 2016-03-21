# coding: utf-8
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import pickle
import os
import time

class RunFastNetwork():
	'''
	用来存储runfast游戏中agent的Q值
	'''
	def __init__(self, name='', inputNum=192, hiddenNum=192, outNum=1):
		self.net = buildNetwork(inputNum, hiddenNum, outNum)
		self.ds = SupervisedDataSet(inputNum, outNum)
		self.name = name
		self.turn = 0

	def train(self, input, output):
		self.ds.clear()
		self.ds.addSample(input, output)
		trainer = BackpropTrainer(self.net, self.ds)
		trainer.train()


	def saveNet(self, filename=''):
		print self.name  + '/' + str(self.turn), ' has saved'
		with open(self.name  + '/' + str(self.turn), 'w') as f:
			pickle.dump(self.net, f)

	def loadNet(self, dir, turn=0):
		print 'loading ', self.name  + '/' + str(turn)
		time.sleep(1)
		if os.path.isfile(self.name  + '/' + str(turn)):
			with open(self.name  + '/' + str(turn), 'r') as f:
				self.net = pickle.load(f)

	def getValue(self, input):
		return self.net.activate(input)

	# def buildNetwork(self, inputNum, hiddenNum, outNum):
	# 	n = FeedForwardNetwork()
	# 	inLayer = LinearLayer(inputNum)
	# 	hiddenLayer = SigmoidLayer(hiddenNum)
	# 	outLayer = LinearLayer(outNum)
	# 	n.addInputModule(inLayer)
	# 	n.addModule(hiddenLayer)
	# 	n.addOutputModule(outLayer)
	# 	in_to_hidden = FullConnection(inLayer, hiddenLayer)
	# 	n.addConnection(in_to_hidden)
	# 	hidden_to_out = FullConnection(hiddenLayer, outLayer)
	# 	n.addConnection(in_to_hidden)
	# 	n.addConnection(hidden_to_out)
	# 	n.sortModules()

	# 	self.layers['in_to_hidden'] = in_to_hidden
	# 	self.layers['hidden_to_out'] = hidden_to_out
	# 	return n
		# self.net = n

class StateNetwork():
	'''
	用来存储状态转移的函数的，具体来说就是我给定一个input，返回给我下一个时刻的状态
	'''
	def __init__(self, name='', inputNum=192, hiddenNum=192, outNum=192):
		self.net = buildNetwork(inputNum, hiddenNum, outNum)
		self.ds = SupervisedDataSet(inputNum, outNum)
		self.name = name
		self.turn = 0

	def train(self, input, output):
		self.ds.clear()
		self.ds.addSample(input, output)
		trainer = BackpropTrainer(self.net, self.ds)
		trainer.train()


	def saveNet(self, filename=''):
		print self.name  + '/' + str(self.turn), ' has saved'
		with open(self.name  + '/' + str(self.turn), 'w') as f:
			pickle.dump(self.net, f)

	def loadNet(self, dir, turn=0):
		print 'loading ', self.name  + '/' + str(turn)
		time.sleep(1)
		if os.path.isfile(self.name  + '/' + str(turn)):
			with open(self.name  + '/' + str(turn), 'r') as f:
				self.net = pickle.load(f)

	def getValue(self, input):
		output = self.net.activate(input)
		for i,v in enumerate(output):
			if v > 0.5:
				output[i] = 1
			else:
				output[i] = 0
		return output

if __name__ == '__main__':
	# rfn = RunFastNetwork()
	# rfn.train([1,2,3], 2)
	# print rfn.getValue([1,2,3,])
	# rfn.saveNet('net1')
	# f = open('net1', 'r')
	# net = pickle.load(f)
	# print net.getValue([1,2,3])
	# sn = StateNetwork()
	# input = [i for i in range(0,192)]
	# print sn.getValue(input)