# coding: utf-8

class Experiment():

	def __init__(self, env, agents):
		self.env = env
		self.agents = agents

	def setAgentsTurn(self, turn):
		for a in self.agents:
			a.setTurn(turn)

	def reset(self):
		self.env.resetEnv()
		for a in self.agents:
			a.reset()

	def doEpisode(self):
		'''
		保存三个agent需要的state
		主要是进行一次episode，然后调用env和agent的接口来更新agent的Q值函数
		主要流程是：选取一个agent走牌，获得agent得到的状态，并根据状态获得action，执行action，这时环境移动到了下一个状态；
				  agent进行学习，更新自己的Q值函数；存储刚才的状态和行动，为下一次learn做准备
		'''
		winner = ''
		agents = self.agents
		env = self.env
		env.doReadyWork(agents)
		while not env.isOver():
			ct = env.currentTurn
			ctagent = agents[ct]
			state = env.getState()
			reward = env.getReward(ctagent)
			action = ctagent.getAction(state)
			env.doAction(action)
			ctagent.learn(state, reward)
			ctagent.lastaction = action['cards']
			ctagent.laststate = state

		for agent in agents:
			state = env.getState()
			playerCards = agent.getCurrentCards()
			state['playerCards'] = playerCards
			if not playerCards:
				winner = agent.name
			reward = env.getReward(agent)
			ctagent.learn(state, reward)

		for agent in agents:
			if agent.controller.turn % 10000 == 0:
				agent.saveNet()

		self.reset()
		print winner, ' wins!'
		return winner

	def doTest(self, testName):
		'''
		具体的进行测试，一个玩家使用网络，另外两个玩家随机
		'''
		print 'start one game'
		agents = self.agents
		env = self.env
		env.doReadyWork(agents)
		winner = ''
		while not env.isOver():
			ct = env.currentTurn
			ctagent = agents[ct]
			state = env.getState()
			action = []
			if ctagent.name == testName:
				action = ctagent.getBestAction(state)
			else:
				action = ctagent.getAction(state)

			env.doAction(action)

		winners = {'player0': None, 'player1': None, 'player2': None, 'name': ''}
		winValue = 0

		for agent in agents:
			playerCards = agent.getCurrentCards()
			print agent.name, playerCards
			if not playerCards:
				winners['name'] = agent.name
			else:
				winners[agent.name] = -len(playerCards)
				winValue += len(playerCards)
		winners[winners['name']] = winValue

		self.reset()
		print winners
		return winners

	def trainState(self, stateNetwork):
		agents = self.agents
		env = self.env
		env.doReadyWork(agents)
		while not env.isOver():
			ct = env.currentTurn
			ctagent = agents[ct]
			state = env.getState()
			action = ctagent.getAction(state)
			env.doAction(action)

		for agent in agents:
			state = env.getState()
			playerCards = agent.getCurrentCards()
			state['playerCards'] = playerCards
			if not playerCards:
				winner = agent.name
			reward = env.getReward(agent)
			ctagent.learn(state, reward)

		for agent in agents:
			if agent.controller.turn % 10000 == 0:
				agent.saveNet()

		env.resetEnv()

