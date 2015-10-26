import numpy as np
import itertools as it

def hood_vec(name):
	if int(name) == 0:
		return [1, 0, 0]
	elif int(name) == 1:
		return [0, 1, 0]
	elif int(name) == 2:
		return [0, 0, 1]

def binary_vec(name):
	if name == 'true':
		return [1]
	else:
		return [0]

def binary_vec_opp(name):
	if name == 'true':
		return [0]
	else:
		return [1]

def valid_params(orders, riders, hour):
	if ((int(orders) >= 0) and (int(orders) <= 28) and (int(riders) >= 0) and 
		(int(riders) <= 17) and (int(hour) >= 11) and (int(hour) <= 21)):
		return True
	else: 
		return False

def predict(model, h_numeric, o_numeric, orders, rn_numeric, riders, hour):
	X = np.concatenate((list([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], 
		[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]), list(it.repeat(h_numeric, 5)), 
		list(it.repeat(o_numeric, 5)), list(it.repeat([int(orders)], 5)), 
		list(it.repeat(rn_numeric, 5)), list(it.repeat([int(riders)], 5)), 
		list(it.repeat([int(hour)], 5))), axis = 1)
	result = model.predict(X)
	return result

def bs_se(y, result):
	for i in np.arange(10000):
		result.append(np.mean(np.random.choice(y, len(y), replace = True)))
	return result
