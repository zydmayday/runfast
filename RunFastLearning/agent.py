# coding: utf-8
__author__ = 'zhangyidong'

from runFast import Player, RunFast
import os, sys
from collections import Counter
import time

NNINPUT = ['14A','15A','3A','4A','5A','6A','7A','8A','9A','10A','11A','12A','13A',
        '14B','3B','4B','5B','6B','7B','8B','9B','10B','11B','12B','13B',
        '14C','3C','4C','5C','6C','7C','8C','9C','10C','11C','12C','13C',
        '3D','4D','5D','6D','7D','8D','9D','10D','11D','12D','13D',
        '14A','15A','3A','4A','5A','6A','7A','8A','9A','10A','11A','12A','13A',
        '14B','3B','4B','5B','6B','7B','8B','9B','10B','11B','12B','13B',
        '14C','3C','4C','5C','6C','7C','8C','9C','10C','11C','12C','13C',
        '3D','4D','5D','6D','7D','8D','9D','10D','11D','12D','13D',
        '14A','15A','3A','4A','5A','6A','7A','8A','9A','10A','11A','12A','13A',
        '14B','3B','4B','5B','6B','7B','8B','9B','10B','11B','12B','13B',
        '14C','3C','4C','5C','6C','7C','8C','9C','10C','11C','12C','13C',
        '3D','4D','5D','6D','7D','8D','9D','10D','11D','12D','13D',
        '14A','15A','3A','4A','5A','6A','7A','8A','9A','10A','11A','12A','13A',
        '14B','3B','4B','5B','6B','7B','8B','9B','10B','11B','12B','13B',
        '14C','3C','4C','5C','6C','7C','8C','9C','10C','11C','12C','13C',
        '3D','4D','5D','6D','7D','8D','9D','10D','11D','12D','13D',]

class RunFastAgent(Player):
    '''
    继承自Player类，用来执行出牌的操作
    controller用来存储Q值，这里实际上是一个神经网络
    通过和环境交互来进行学习，训练出一个收敛的神经网络
    函数都是通过experiment进行调用
    '''

    def __init__(self, name, controller, alpha=0.5, gamma=0.9):
        '''
        存储laststate，以便在新的state到来的时候更新controller
        '''
        Player.__init__(self, name)
        self.controller = controller
        self.laststate = None 
        self.lastaction = None
        self.alpha = alpha
        self.gamma = gamma

    def setTurn(self, turn):
        self.controller.turn = turn

    def reset(self):
        self.laststate = None 
        self.lastaction = None

    def getAction(self, state):
        '''
        根据当前的牌面状态选择一个action
        return dict{cards，type}
        '''
        preCards = state['preCards']
        preType = state['preType']
        if state['isFirst']:
            return self.playCardsWithHart3()
        if preCards:
            return self.playRandomByPreCards(preType, preCards)
        else:
            return self.playRandom()

    def getActions(self, state):
        '''
        找出当前状态下所有可能的打法
        '''
        preCards = state['preCards']
        preType = state['preType']
        isFirst = state['isFirst']
        cardsCanPlay = self.getCardsCanPlay(preType, preCards, isFirst)
        return cardsCanPlay

    def getBestAction(self, state):
        '''
        这里有几种方法来实现，先说最简单的。
        直接对当前的状态s取各个可能的action的Q值，max的那个a就可以作为best
        '''
        actionDict = self.getActions(state)
        # time.sleep(1)
        # print actionDict
        bestAction = []
        bestType = None
        bestValue = None
        for naType in actionDict.keys():
            # print naType
            for a in actionDict[naType]:
                # print a
                inp = self.getInput(state, a)
                value = self.controller.getValue(inp)
                if not bestValue:
                    bestValue = value
                    bestType = naType
                    bestAction = a
                if  value > bestValue:
                    bestValue = value
                    bestType = naType
                    bestAction = a

        playCards = [self.cards[i] for i in bestAction]
        return {'cards': playCards, 'type': bestType}


    def getInput(self, state, action):
        '''
        获得当前的状态，供网络学习
        playerCards 当前的手牌
        playedCards 已经打出的牌
        preCards 上家打出的牌
        actions 这一轮的走牌
        '''
        input = [0 for i in range(0, 192)]
        playerCards = state['playerCards']
        playedCards = state['playedCards']
        preCards = state['preCards']
        for i, c in enumerate(NNINPUT):
            if i < 48 and c in playerCards:
                input[i] = 1
            elif 48 <= i < 96 and c in playedCards:
               input[i] = 1
            elif 96 <= i < 144 and c in preCards:
                input[i] = 1   
            elif 144 <= i < 192  and c in action:
                input[i] = 1
        return input

    def learn(self, nextState, reward):
        '''
        算出target Qvalue，传给控制器进行学习
        qvalue_n+1 = (1-alpha) * qvalue + alpha*(r + gamma * max(qvalue_next))
        怎么处理最后一次优化：最后一次时，我们获得了终盘的状态和回报，这时候，应该没有max（）的这一项了，因为没有action可走了，所有去掉这一项
        '''

        if not self.lastaction or not self.laststate:
            return False
        # print self.laststate
        input = self.getInput(self.laststate, self.lastaction)
        qValue = self.controller.getValue(input)

        qNextValues = []
        nextActionsDict = self.getActions(nextState)
        for naType in nextActionsDict.keys():
            for a in nextActionsDict[naType]:
                inp = self.getInput(nextState, a)
                qNextValues.append(self.controller.getValue(inp))
        maxQ = 0
        if qNextValues:
            maxQ = max(qNextValues)
        tagetValue = (1 - self.alpha) * qValue + self.alpha * (reward + self.gamma * maxQ)

        self.controller.train(input, tagetValue)
        # 如果没有下一个value的话，就说明我们的这个episode结束了，我们就把训练好的神经网络存入文件中
        # if not qNextValues:
        #     if not os.path.isdir(self.name):
        #         os.mkdir(self.name)
        #     self.controller.saveNet(self.name + '/net' + str(self.turn))

    def saveNet(self):
        if not os.path.isdir(self.controller.name):
            os.mkdir(self.controller.name)
        self.controller.saveNet()


