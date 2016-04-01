# coding:utf-8
from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent
from controller import RunFastNetwork, RunFastDeepNetwork, StateNetwork
import pickle
import os
from collections import defaultdict
import time

def trainQValueNetwork(loopNum=1000000, startTurn=0, playerNamePrefix='player', history_filename='train_winners_nn'):
	'''
	通过让三个agent互相玩游戏，然后来训练出一个Q值网络
	三个agent的网络保存在playeri里面，数字分别代表的是训练了多少次后得出的网络
	胜负情况记录在train_winners里面
	'''
	nws = []
	agents = []
	winners = {}
	if os.path.isfile(history_filename):
		with open(history_filename, 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()]) 

	for i in range(0, 3):
		playerName = playerNamePrefix + str(i)
		nw = RunFastNetwork(playerName)
		nw.loadNet(playerName, startTurn)
		rfa = RunFastAgent(playerName, nw)
		nws.append(nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents)

	for i in range(startTurn, startTurn + loopNum):
		exp.setTurn(i)
		winner = exp.doEpisode()
		if winners.has_key(winner):
			winners[winner] += 1
		else:
			winners[winner] = 1

	print winners
	with open(history_filename, 'w') as f:
		pickle.dump(winners, f)

def testQValueNetwork(startTurn=0, loopNum=1000, testName='player0', filename='test_winners_nn', playerNamePrefix='player'):
	'''
	在测试时，其他的两个agent都不选用最佳的network，只有测试对象使用
	然后测试对象每次选取最佳的行动，其他的两个agent有50%概率选择最佳行动，不过他们的net应该是最普通的，没有经过训练的
	'''
	agents = []
	winNums = {}
	if os.path.isfile(filename):
		with open(filename, 'r') as f:
			winNums = pickle.load(f)

	print 'loading agents'
	for i in range(0, 3):
		playerName = playerNamePrefix + str(i)
		nw = RunFastNetwork(playerName)
		if playerName == testName:
			nw.loadNet(playerName, startTurn)
		rfa = RunFastAgent(playerName, nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents)

	print 'set up the experiment'

	for i in range(startTurn, startTurn + loopNum):
		if not winNums.get(startTurn):
			winNums[startTurn] = {}
		testHistory = exp.doTest(testName)
		for j in range(0,3):
			playerName = playerNamePrefix + str(j)
			if not winNums[startTurn].get(playerName):
				winNums[startTurn][playerName] = testHistory[playerName]
			else:
				winNums[startTurn][playerName] += testHistory[playerName]
		print str(i-startTurn), winNums

	print winNums
	with open(filename, 'w') as f:
		pickle.dump(winNums, f)

if __name__ == '__main__':
	train = input('input 1 to train, input 0 to test:')
	playerNamePrefix = 'player_nn'
	if train:
		trainQValueNetwork(playerNamePrefix=playerNamePrefix)
	else:
		testName = playerNamePrefix + '0'
		for i in range(0,1000000,20000):
			while not os.path.isfile(testName + '/' + str(i)):
				print 'waiting for training finish'
				time.sleep(10)
			testQValueNetwork(startTurn=i, loopNum=100000, filename='test_winners_nn', playerNamePrefix=playerNamePrefix, testName=testName)
