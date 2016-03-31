# coding:utf-8
from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent
from controller import RunFastNetwork, RunFastDeepNetwork, StateNetwork
import pickle
import os
from collections import defaultdict
import time


def trainDeepNetwork(loopNum=1000000, startTurn=0, playerNamePrefix='player', history_filename='train_winners_dn'):
	'''
	用深度网络来训练Q值
	'''
	nws = []
	agents = []
	winners = {}

	# load history match
	if os.path.isfile(history_filename):
		with open(history_filename, 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()]) + 1

	# load agents with network
	for i in range(0, 3):
		playerName = playerNamePrefix + str(i)
		nw = RunFastDeepNetwork(playerName)
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
	with open(history_filename, 'w') as f:
		pickle.dump(winners, f)

def testQValueNetwork(startTurn=0, loopNum=1000, testName='player0', filename='win_nums', playerNamePrefix='player'):
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
	train = input('input 1 to train, input 0 to test:')
	playerNamePrefix = 'player_dn'
	if train:
		trainDeepNetwork(playerNamePrefix=playerNamePrefix)
	else:
		testName = playerNamePrefix + '0'
		for i in range(0,1000000,20000):
			while not os.path.isfile(testName + '/' + str(i)):
				print 'waiting for training finish'
				time.sleep(10)
			testQValueNetwork(startTurn=i, loopNum=100000, filename='test_winners_dn', playerNamePrefix=playerNamePrefix, testName=testName)
