# coding:utf-8
from experiment import Experiment, ExperimentWithMemory
from environment import RunFastEnvironment
from agent import RunFastAgent, RunFastAgentWithMemory
from controller import RunFastNetwork, RunFastDeepNetwork
import pickle
import os
from collections import defaultdict
import time

'''
define some static values
because we have to compare some differences between algorithms
since then, we do not need to create a lot of files that we can do things in only one py,
by passing the input to the main function
'''
PLAYER_LIST = {'dn': ['dn_train1', 'dn_train2', 'dn_train3'], 'nn': ['nn_train1', 'nn_train2', 'nn_train3'], 'nn_wm': ['nn_wm_train1', 'nn_wm_train2', 'nn_wm_train3'], 'dn_wm': ['dn_wm_train1', 'dn_wm_train2', 'dn_wm_train3']}
NETWORK = {'dn': RunFastDeepNetwork, 'dn_wm': RunFastDeepNetwork, 'nn': RunFastNetwork, 'nn_wm': RunFastNetwork}
AGENT = {'dn': RunFastAgent, 'dn_wm': RunFastAgentWithMemory, 'nn': RunFastAgent, 'nn_wm': RunFastAgentWithMemory}
EXPERIMENT = {'dn': Experiment, 'dn_wm': ExperimentWithMemory, 'nn': Experiment, 'nn_wm': ExperimentWithMemory}
TRAIN = {'dn': 'train_dn.dict', 'dn_wm': 'train_dn_wm.dict', 'nn': 'train_nn.dict', 'nn_wm': 'train_nn_wm.dict'}
TEST = {'dn': 'test_dn.dict', 'dn_wm': 'test_dn_wm.dict', 'nn': 'test_nn.dict', 'nn_wm': 'test_nn_wm.dict'}

def trainDeepNetwork(loopNum=10000, startTurn=0, type='nn'):
	'''
	用深度网络来训练Q值
	'''
	agents = []
	winners = {}
	train_filename = TRAIN[type]
	# load history match
	if os.path.isfile(train_filename):
		with open(train_filename, 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()]) 

	# load agents with network
	for i in range(0, 3):
		playerName = PLAYER_LIST[type][i]
		nw = NETWORK[type](playerName)
		nw.loadNet(playerName, startTurn)
		rfa = AGENT[type](playerName, nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = EXPERIMENT[type](env, agents)

	for i in range(startTurn, startTurn + loopNum):
		if i % 100 == 0:
			for agent in agents:
				agent.saveNet()
			with open(train_filename, 'w') as f:
				pickle.dump(winners, f)
		winner = exp.doEpisode()
		if winners.has_key(winner):
			winners[winner] += 1
		else:
			winners[winner] = 1
	for agent in agents:
		agent.saveNet()
	with open(train_filename, 'w') as f:
		pickle.dump(winners, f)

def testQValueNetwork(startTurn=0, loopNum=1000, type=''):
	agents = []
	win_nums = {}
	test_name = PLAYER_LIST[type][0]
	test_filename = TEST[type]
	if os.path.isfile(test_filename):
		with open(test_filename, 'r') as f:
			win_nums = pickle.load(f)

	for i in range(0, 3):
		playerName = PLAYER_LIST[type][i]
		nw = NETWORK[type](playerName)
		if playerName == test_name:
			nw.loadNet(playerName, startTurn)
		rfa = AGENT[type](playerName, nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents)

	for i in range(startTurn, startTurn + loopNum):
		if not win_nums.get(startTurn):
			win_nums[startTurn] = {}
		testHistory = exp.doTest(test_name)
		for j in range(0,3):
			playerName = PLAYER_LIST[type][j]
			if not win_nums[startTurn].get(playerName):
				win_nums[startTurn][playerName] = {'point': 0, 'win': 0}
			win_nums[startTurn][playerName]['point'] += testHistory[playerName]
			if testHistory['name'] == playerName:
				win_nums[startTurn][playerName]['win'] += 1
	with open(test_filename, 'w') as f:
		pickle.dump(win_nums, f)

if __name__ == '__main__':
	train = input('input 1 to train, input 0 to test: ')
	type = raw_input('input the TYPE you want to train/test: ')
	if train:
		trainDeepNetwork(type=type, loopNum=10000)
	else:
		test_filename = TEST[type]
		test_name = PLAYER_LIST[type][0]
		winNums = {0:{}}
		if os.path.isfile(test_filename):
			with open(test_filename, 'r') as f:
				winNums = pickle.load(f)
		startTurn = max(winNums.keys())
		for i in range(startTurn, startTurn + 10000, 100):
			while not os.path.isfile(test_name + '/' + str(i)):
				print 'waiting for training finish', i
				time.sleep(10)
			testQValueNetwork(startTurn=i, loopNum=10000, type=type)
