# coding:utf-8
from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent
from controller import RunFastNetwork
import pickle
import os
from collections import defaultdict

def trainQValueNetwork(loopNum=1000000, startTurn=0):
	'''
	通过让三个agent互相玩游戏，然后来训练出一个Q值网络
	'''
	nws = []
	agents = []
	winners = {}
	if os.path.isfile('train_winners'):
		with open('train_winners', 'r') as f:
			winners = pickle.load(f)
			startTurn = sum([v for i,v in winners.items()]) + 1

	for i in range(0, 3):
		nw = RunFastNetwork('player' + str(i))
		nw.loadNet('player' + str(i), startTurn)
		rfa = RunFastAgent('player' + str(i), nw)
		nws.append(nw)
		agents.append(rfa)
		 
	for k in winners.keys():
		startTurn += winners[k]

	env = RunFastEnvironment()
	exp = Experiment(env, agents)

	for i in range(startTurn, startTurn + loopNum):
		exp.setAgentsTurn(i)
		winner = exp.doEpisode()
		if winners.has_key(winner):
			winners[winner] += 1
		else:
			winners[winner] = 1

		if i % 10000 == 0:
			with open('train_winners', 'w') as f:
				pickle.dump(winners, f)

	print winners
	with open('train_winners', 'w') as f:
		pickle.dump(winners, f)

def trainStateTransitionNetwork():
	pass

def testQValueNetwork(startTurn=0, loopNum=1000, testName='player0'):
	'''
	其中一个玩家使用训练好的网络，其他两个agent随机出牌，记录胜率
	winNums = {20000: 57777, 40000:69999,...}
	'''
	agents = []
	winNums = defaultdict(int)
	if os.path.isfile('win_nums'):
		with open('win_nums', 'r') as f:
			winNums = pickle.load(f)

	print 'loading agents'
	for i in range(0, 3):
		nw = RunFastNetwork('player' + str(i))
		nw.loadNet('player' + str(i), startTurn)
		rfa = RunFastAgent('player' + str(i), nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents)

	print 'set up the experiment'

	for i in range(startTurn, startTurn + loopNum):
		winners = exp.doTest(testName)
		winNums[startTurn] += winners[testName]
		print testName, ' got (', winNums[startTurn], '/', i-startTurn, ')'

	print winNums
	with open('win_nums', 'w') as f:
		pickle.dump(winNums, f)

if __name__ == '__main__':
	# trainQValueNetwork()
	for i in range(0,1000000,10000):
		testQValueNetwork(startTurn=i, loopNum=10000)
	# testQValueNetwork(startTurn=0, loopNum=10)
