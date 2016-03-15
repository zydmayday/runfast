from experiment import Experiment
from environment import RunFastEnvironment
from agent import RunFastAgent
from controller import RunFastNetwork

def main():
	nws = []
	agents = []
	winners = {}
	for i in range(0, 3):
		nw = RunFastNetwork()
		rfa = RunFastAgent('player' + str(i), nw)
		nws.append(nw)
		agents.append(rfa)
		 
	env = RunFastEnvironment()
	exp = Experiment(env, agents)
	for i in range(0, 5):
		exp.setAgentsTurn(i)
		winner = exp.doEpisode()
		if winners.has_key(winner):
			winners[winner] += 1
		else:
			winners[winner] = 0

	print winners

if __name__ == '__main__':
	main()