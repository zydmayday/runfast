# coding:utf-8
from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent
from controller import RunFastNetwork, RunFastDeepNetwork, StateNetwork
import pickle
import os
from collections import defaultdict
import time

PLAYER_LIST = ['nn_train1', 'nn_train2', 'nn_train3']

def trainQValueNetwork(loopNum=10000, startTurn=0, history_filename='train_winners_nn', inputNum=192, type=1):
	'''
	通过让三个agent互相玩游戏，然后来训练出一个Q值网络
	三个agent的网络保存在playeri里面，数字分别代表的是训练了多少次后得出的网络
	胜负情况记录在train_winners里面
	'''
	agents = []
	winners = {}
	if os.path.isfile(history_filename):
		with open(history_filename, 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()]) 

	for i in range(0, 3):
		playerName = PLAYER_LIST[i]
		nw = RunFastNetwork(playerName, inputNum=inputNum, hiddenNum=inputNum, outNum=1)
		nw.loadNet(playerName, startTurn)
		rfa = RunFastAgent(playerName, nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents, type=type)

	for i in xrange(startTurn, startTurn + loopNum):

		if i % 200 == 0:
			for agent in agents:
				agent.saveNet()

			with open(history_filename, 'w') as f:
				pickle.dump(winners, f)
		# exp.setTurn(i)
		winner = exp.doEpisode()
		if winners.has_key(winner):
			winners[winner] += 1
		else:
			winners[winner] = 1

	for agent in agents:
		agent.saveNet()

	with open(history_filename, 'w') as f:
		pickle.dump(winners, f)


	print winners
	with open(history_filename, 'w') as f:
		pickle.dump(winners, f)

def testQValueNetwork(startTurn=0, loopNum=1000, testName='player0', filename='test_winners_nn', playerNamePrefix='player', type=1, inputNum=192):
	'''
	在测试时，其他的两个agent都不选用最佳的network，只有测试对象使用
	然后测试对象每次选取最佳的行动，其他的两个agent有50%概率选择最佳行动，不过他们的net应该是最普通的，没有经过训练的
	winNums = {trainNum: {player0: {'point': xxx, 'win': yyy}, player1: {...}}}
	'''
	agents = []
	winNums = {}
	if os.path.isfile(filename):
		with open(filename, 'r') as f:
			winNums = pickle.load(f)

	print 'loading agents'
	for i in range(0, 3):
		playerName = PLAYER_LIST[i]
		nw = RunFastNetwork(playerName, inputNum=inputNum, hiddenNum=inputNum, outNum=1)
		if playerName == testName:
			nw.loadNet(testName, startTurn)
		rfa = RunFastAgent(playerName, nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents, type=type)

	print 'set up the experiment'

	for i in range(startTurn, startTurn + loopNum):
		if not winNums.get(startTurn):
			winNums[startTurn] = {}
		testHistory = exp.doTest(testName)
		for j in range(0,3):
			playerName = PLAYER_LIST[j]
			if not winNums[startTurn].get(playerName):
				winNums[startTurn][playerName] = {'point': 0, 'win': 0}
			winNums[startTurn][playerName]['point'] += testHistory[playerName]
			if testHistory['name'] == playerName:
				winNums[startTurn][playerName]['win'] += 1
		print str(i-startTurn), winNums

	print winNums
	with open(filename, 'w') as f:
		pickle.dump(winNums, f)

def main1():
	train = input('input 1 to train, input 0 to test:')
	if train:
		trainQValueNetwork()
	else:
		testName = PLAYER_LIST[0]
		testFileName = 'test_winners_nn'
		winNums = {0:{}}
		if os.path.isfile(testFileName):
			with open(testFileName, 'r') as f:
				winNums = pickle.load(f)
		startTurn = max(winNums.keys())
		for i in range(startTurn + 200, startTurn + 10000, 200):
			while not os.path.isfile(testName + '/' + str(i)):
				print 'not found ', testName + '/' + str(i) ,'waiting for training finish'
				time.sleep(10)
			testQValueNetwork(startTurn=i, loopNum=10000, filename=testFileName, testName=testName)

def main2():
	train = input('input 1 to train, input 0 to test:')
	if train:
		trainQValueNetwork(inputNum=52, loopNum=1, type=2)
	else:
		testName = PLAYER_LIST[0]
		for i in range(0,1000000,5000):
			while not os.path.isfile(testName + '/' + str(i)):
				print 'not found ', testName + '/' + str(i) ,'waiting for training finish'
				time.sleep(10)
			testQValueNetwork(startTurn=i, loopNum=50000, filename='test_winners_nn', testName=testName)

if __name__ == '__main__':
	main1()
	# main2()
