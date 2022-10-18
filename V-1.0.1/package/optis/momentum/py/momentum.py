from kernel.py.optis import Opti_Patern

class MOMENTUM(Opti_Patern):
	name = "MOMENTUM"

	description = '''	v = moment * v - alpha * grad(w)
	w += v'''

	CONSTS = {
		'ALPHA' : 1e-5,
		'MOMENT' : 1e-4
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

				dw = self.train._meand[wpos] / lines
				self.hist[wpos] = self.CONSTS['MOMENT']*self.hist[wpos] - self.CONSTS['ALPHA']*dw

				self.train._weight[wpos] += self.hist[wpos]