# coding:utf-8
from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent, RunFastAgentWithMemory
from controller import RunFastNetwork, RunFastDeepNetwork
import pickle
import os
from collections import defaultdict
import time

# PLAYER_LIST = ['dn_memory_train1', 'dn_memory_train2', 'dn_memory_train3']
PLAYER_LIST = ['dn_memory_train1_1000', 'dn_memory_train2_1000', 'dn_memory_train3_1000']

def trainDeepNetworkWithMemory(loopNum=30000, startTurn=0, history_filename='train_winners_dn_with_memory_1000', inputNum=192, type=1):
	'''
	使用带记忆的方式来训练深度神经网络
	'''
	agents = []
	winners = {}

	# load history match
	if os.path.isfile(history_filename):
		with open(history_filename, 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()])
	print startTurn
	# load agents with network
	for i in range(0, 3):
		playerName = PLAYER_LIST[i]
		nw = RunFastDeepNetwork(playerName, inputNum=inputNum, hidden1Num=inputNum, hidden2Num=inputNum, hidden3Num=inputNum, outNum=1)
		nw.loadNet(playerName, startTurn)
		rfa = RunFastAgentWithMemory(playerName, nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents, type=type)

	for i in range(startTurn, startTurn + loopNum):
		# exp.setTurn(i)
		if i % 200 == 0:
			for agent in agents:
				agent.saveNet()
			with open(history_filename, 'w') as f:
				pickle.dump(winners, f)

		winner = exp.doEpisodeWithMemory(capacity=1000)
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

def testQValueNetwork(startTurn=0, loopNum=1000, testName='player0', filename='win_nums', type=1, inputNum=192):
	agents = []
	winNums = {}
	if os.path.isfile(filename):
		with open(filename, 'r') as f:
			winNums = pickle.load(f)

	print 'loading agents'
	for i in range(0, 3):
		playerName = PLAYER_LIST[i]
		nw = RunFastDeepNetwork(playerName, inputNum=inputNum, hidden1Num=inputNum, hidden2Num=inputNum, hidden3Num=inputNum, outNum=1)
		if playerName == testName:
			nw.loadNet(playerName, startTurn)
		rfa = RunFastAgentWithMemory(playerName, nw)
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

if __name__ == '__main__':
	train = input('input 1 to train, input 0 to test:')
	if train:
		trainDeepNetworkWithMemory()
	else:
		testName = PLAYER_LIST[0]
		testFileName = 'test_winners_dn_with_memory_1000'
		# testFileName = 'test_winners_dn_with_memory'
		winNums = {0:{}}
		if os.path.isfile(testFileName):
			with open(testFileName, 'r') as f:
				winNums = pickle.load(f)
		startTurn = max(winNums.keys())
		for i in range(startTurn, startTurn + 30000, 200):
			while not os.path.isfile(testName + '/' + str(i)):
				print 'waiting for training finish'
				time.sleep(10)
			testQValueNetwork(startTurn=i, loopNum=10000, filename=testFileName, testName=testName)
