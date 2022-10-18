from kernel.py.optis import Opti_Patern

class RMSPROP(Opti_Patern):
	name = "RMSPROP"

	description = '''v = beta * v + (1-beta) * grad(w)^2
	w -= alpha * grad(w) / sqrt(v)'''

	CONSTS = {
		'ALPHA' : 1e-5,
		'BETA' : 1e-4 
	}

	MIN_TEST_ECHOPES = 2

	def __init__(self, opti):
		self.train = opti.train

		self.hist = [0 for i in range(opti.train.model.weights * opti.train.sets)]

	def __del__(self):
		del self.hist

	def opti(self):
		mdl = self.train.model
		sets = self.train.sets
		ws = mdl.weights
		lines = self.train.data.lines

		for s in range(sets):
			for w in range(ws):
				wpos = s*ws + w

				dw = self.train._meand[s*ws + w] / lines
				self.hist[wpos] = self.CONSTS['BETA']*self.hist[wpos] + (1 - self.CONSTS['BETA'])*dw**2

				self.train._weight[wpos] -= self.CONSTS['ALPHA'] * dw * (self.hist[wpos] + 1e-8)**(-.5)