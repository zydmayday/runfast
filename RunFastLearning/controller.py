# coding: utf-8
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
import pickle

class RunFastNetwork():
	'''
	用来存储runfast游戏中agent的Q值
	'''
	def __init__(self, inputNum=192, hiddenNum=192, outNum=1):
		self.layers = {'in_to_hidden': [], 'hidden_to_out': []}
		self.net = buildNetwork(inputNum, hiddenNum, outNum)
		self.ds = SupervisedDataSet(inputNum, outNum)

	def train(self, input, output):
		self.ds.clear()
		self.ds.addSample(input, output)
		trainer = BackpropTrainer(self.net, self.ds)
		trainer.train()


	def saveNet(self, filename):
		with open(filename, 'w') as f:
			pickle.dump(self, f)

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

if __name__ == '__main__':
	rfn = RunFastNetwork()
	rfn.train([1,2,3], 2)
	print rfn.getValue([1,2,3,])
	rfn.saveNet('net1')
	f = open('net1', 'r')
	net = pickle.load(f)
	print net.getValue([1,2,3])

