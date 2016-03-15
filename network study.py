from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork

if __name__ == '__main__':

	n = FeedForwardNetwork()
	inLayer = LinearLayer(2)
	hiddenLayer = SigmoidLayer(3)
	outLayer = LinearLayer(1)

	n.addInputModule(inLayer)
	n.addModule(hiddenLayer)
	n.addOutputModule(outLayer)


	in_to_hidden = FullConnection(inLayer, hiddenLayer)
	hidden_to_out = FullConnection(hiddenLayer, outLayer)

	n.addConnection(in_to_hidden)
	n.addConnection(hidden_to_out)

	n.sortModules()

	print n.activate([1,2])

	n2 = RecurrentNetwork()
	n2.addInputModule(LinearLayer(2, name='in'))
	n2.addModule(SigmoidLayer(3, name='hidden'))
	n2.addOutputModule(LinearLayer(1, name='out'))
	n2.addConnection(FullConnection(n2['in'], n2['hidden'], name='c1'))
	n2.addConnection(FullConnection(n2['hidden'], n2['out'], name='c2'))
	n2.addRecurrentConnection(FullConnection(n2['hidden'], n2['hidden'], name='c3'))
	n2.sortModules()

	print n.activate((2,2))
	print n.activate((2,2))
	print n.activate((2,2))