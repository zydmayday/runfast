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

	def doOneTurn(self):
		'''
		先获得新得到的状态和回报值，然后根据新的状态选择action，再执行动作
		根据上一次的执行动作和状态，以及这一次获得的回报值和新的状态来更新Q值函数
		'''
		env = self.env
		agents = self.agents
		ct = env.currentTurn
		ctagent = agents[ct]
		state = env.getState()
		lastreward = env.getReward(ctagent)
		action = ctagent.getAction(state)
		ctagent.learn(state, lastreward)
		ctagent.lastaction = action['cards']
		ctagent.laststate = state
		env.doAction(action)

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
			self.doOneTurn()
		else:
			for agent in agents:
				playerCards = agent.getCurrentCards()
				reward = env.getReward(agent)
				agent.learn(None, reward)
				if not playerCards:
					winner = agent.name

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
		Experiment.__init__(self, env, agents)
		self.capacity = capacity

	def doOneTurnWithMemory(self, capacity=10000):
		env = self.env
		agents = self.agents
		ct = env.currentTurn
		ctagent = agents[ct]
		state = env.getState()
		lastreward = env.getReward(ctagent)
		action = ctagent.getAction(state)
		ctagent.saveMemory(lastreward, state, action, capacity=capacity)
		ctagent.learnFromMemory()
		env.doAction(action)

	def doEpisode(self):
		'''
		存储历史纪录，不使用及时的更新，而是从历史纪录中调取纪录进行更新
		'''
		winner = ''
		agents = self.agents
		env = self.env
		env.doReadyWork(agents)
		capacity  = self.capacity
		while not env.isOver():
			self.doOneTurnWithMemory(capacity=self.capacity)
		else:
			for agent in agents:
				lastreward = env.getReward(agent)
				agent.saveMemory(lastreward, None, capacity=capacity)
				agent.learnFromMemory()
				playerCards = agent.getCurrentCards()
				if not playerCards:
					winner = agent.name

		self.addTurn()
		self.reset()
		return winner