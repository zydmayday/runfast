import pickle
if __name__ == '__main__':
	n1Name = raw_input('input net1 name: ')
	n2Name = raw_input('input net2 name: ')
	n1 = pickle.load(open(n1Name))
	n2 = pickle.load(open(n2Name))
	input = [1] * 192
	print n1.activate(input)
	print n2.activate(input)
