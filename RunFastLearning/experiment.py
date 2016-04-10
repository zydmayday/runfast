# coding: utf-8

class Experiment():

	def __init__(self, env, agents, type=1):
		self.env = env
		self.agents = agents
		self.type = type

	def setTurn(self, turn):
		for a in self.agents:
			a.setTurn(turn)

	def addTurn(self):
		for a in self.agents:
			a.controller.turn += 1		

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
		type = self.type
		while not env.isOver():
			ct = env.currentTurn
			ctagent = agents[ct]

			state = env.getState()
			reward = env.getReward(ctagent)
			action = ctagent.getAction(state, type=type)
			env.doAction(action)
			print ctagent.name, 'starting learning'
			ctagent.learn(state, reward, type=type)
			ctagent.lastaction = action['cards']
			ctagent.laststate = state

		for agent in agents:
			state = env.getState()
			playerCards = agent.getCurrentCards()
			state['playerCards'] = playerCards
			if not playerCards:
				winner = agent.name
			reward = env.getReward(agent)
			# print agent.name, reward
			agent.learn(state, reward, type=type)

		self.addTurn()

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
			action = ctagent.getBestAction(state)
			env.doAction(action)

		testHistory = {'player0': None, 'player1': None, 'player2': None, 'name': ''}
		winValue = 0

		for agent in agents:
			playerCards = agent.getCurrentCards()
			# print agent.name, playerCards
			if not playerCards:
				testHistory['name'] = agent.name
			else:
				testHistory[agent.name] = -len(playerCards)
				winValue += len(playerCards)
		testHistory[testHistory['name']] = winValue

		self.reset()
		# print testHistory
		return testHistory

class ExperimentWithMemory(Experiment):

	def __init__(self, env, agents, type=1, capacity=10000):
		Experiment.__init__(env, agents)
		self.capacity = capacity

	def doEpisode(self):
		'''
		存储历史纪录，不使用及时的更新，而是从历史纪录中调取纪录进行更新
		'''
		winner = ''
		agents = self.agents
		env = self.env
		env.doReadyWork(agents)
		capacity  = self.capacity
		type = self.type

		while not env.isOver():
			ct = env.currentTurn
			ctagent = agents[ct]
			state = env.getState()
			reward = env.getReward(ctagent)
			action = ctagent.getAction(state, type=type)
			env.doAction(action)
			# memory = [ctagent.laststate, ctagent.lastaction, reward, state]
			ctagent.saveMemory(reward, state, action, capacity=capacity)
			ctagent.learnFromMemory(type=type)

		for agent in agents:
			state = env.getState()
			playerCards = agent.getCurrentCards()
			state['playerCards'] = playerCards
			if not playerCards:
				winner = agent.name
			reward = env.getReward(agent)
			# memory = [agent.laststate, agent.lastaction, reward, state]
			agent.saveMemory(reward, state, capacity=capacity)
			agent.learnFromMemory(type=type)

		# for agent in agents:
		# 	if agent.controller.turn % 5000 == 0:
		# 		agent.saveNet()

		self.addTurn()

		self.reset()
		return winner