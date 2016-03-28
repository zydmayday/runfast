# coding:utf-8
from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent
from controller import RunFastNetwork, RunFastDeepNetwork, StateNetwork
import pickle
import os
from collections import defaultdict

def trainQValueNetwork(loopNum=1000000, startTurn=0):
	'''
	通过让三个agent互相玩游戏，然后来训练出一个Q值网络
	三个agent的网络保存在playeri里面，数字分别代表的是训练了多少次后得出的网络
	胜负情况记录在train_winners里面
	'''
	nws = []
	agents = []
	winners = {}
	if os.path.isfile('train_winners'):
		with open('train_winners', 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()]) + 1

	for i in range(0, 3):
		playerName = 'player' + str(i)
		nw = RunFastNetwork(playerName)
		nw.loadNet(startTurn)
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
	with open('train_winners', 'w') as f:
		pickle.dump(winners, f)

def trainDeepNetwork(loopNum=1000000, startTurn=0):
	'''
	用深度网络来训练Q值
	结果保存在deep_playeri里面，数字为训练的次数
	胜负结果保存在deep_train_winners里面
	'''
	nws = []
	agents = []
	winners = {}

	# load history match
	if os.path.isfile('deep_train_winners'):
		with open('deep_train_winners', 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()]) + 1

	# load agents with network
	for i in range(0, 3):
		playerName = 'deep_player' + str(i)
		nw = RunFastDeepNetwork(playerName)
		nw.loadNet(startTurn)
		rfa = RunFastAgent(playerName, nw)
		nws.append(nw)
		agents.append(rfa)
		 
	# load state network
	stateName = 'deep_state'
	stateNetwork = StateNetwork(stateName)
	if os.path.isfile(stateName):
		with open(stateName + '/' + str(startTurn), 'r') as f:
			stateNetwork = pickle.load(f)

	env = RunFastEnvironment()
	exp = Experiment(env, agents, stateNetwork)

	for i in range(startTurn, startTurn + loopNum):
		exp.setTurn(i)
		winner = exp.doEpisode()
		if winners.has_key(winner):
			winners[winner] += 1
		else:
			winners[winner] = 1

	print winners
	with open('deep_train_winners', 'w') as f:
		pickle.dump(winners, f)

def trainStateTransitionNetwork():
	pass

def testQValueNetwork(startTurn=0, loopNum=1000, testName='player0', filename='win_nums', playerNamePrefix='player'):
	'''
	其中一个玩家使用训练好的网络，其他两个agent随机出牌，记录胜率
	winNums = {20000: 57777, 40000:69999,...}
	startTurn表示训练的神经网络的次数，loopNum表示test的次数
	'''
	agents = []
	winNums = {}
	if os.path.isfile(filename):
		with open(filename, 'r') as f:
			winNums = pickle.load(f)

	print 'loading agents'
	for i in range(0, 3):
		playerName = playerNamePrefix + str(i)
		nw = RunFastDeepNetwork(playerName)
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
	# trainQValueNetwork()
	# trainDeepNetwork()
	for i in range(0,1000000,10000):
		testQValueNetwork(startTurn=i, loopNum=100000, filename='deep_win_nums', playerNamePrefix='deep_player', testName='deep_player0')
	# testQValueNetwork(startTurn=10000000, loopNum=100, testName='player1')
