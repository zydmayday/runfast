from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent
from controller import RunFastNetwork
import pickle
import os

def main():
	nws = []
	agents = []
	winners = {}
	startTurn = 0
	loopNum = 20

	if os.path.isfile('winners'):
		with open('winners', 'r') as f:
			winners = pickle.load(f)
	for i in range(0, 3):
		nw = RunFastNetwork('player' + str(i))
		nw.loadNet('player' + str(i), 10)
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

	print winners
	with open('winners', 'w') as f:
		pickle.dump(winners, f)

if __name__ == '__main__':
	main()