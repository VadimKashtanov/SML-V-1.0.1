from kernel.py.optis import Opti_Patern

class ADAM(Opti_Patern):
	name = "ADAM"

	description = '''m = beta0*m + (1 - beta0)*grad(w)
v = beta1*m + (1 - beta1)*grad(w)^2

_m = m / ( 1 - beta0^t )
_v = v / ( 1 - beta1^t )		t is echope

w -= alpha * _m / sqrt(_v + eta)'''

	CONSTS = {
		'ALPHA' : 1e-5,
		'BETA0' : 1e-5,
		'BETA1' : 1e-5
	}

	MIN_TEST_ECHOPES = 2

	def __init__(self, opti):
		self.train = opti.train

		self.S = [0 for i in range(opti.train.model.weights * opti.train.sets)]
		self.V = [0 for i in range(opti.train.model.weights * opti.train.sets)]

		self.echopes = 0

	def __del__(self):
		del self.S, self.V #Il me semble que ca va virer la reference de .S et .V et donc ces listes auront 
							#plus aucunes references donc -> garbbage collector

	def opti(self):
		mdl = self.train.model
		sets = self.train.sets
		ws = mdl.weights
		lines = self.train.data.lines

		#	Echopes need for _s/_v
		self.echopes += 1

		for _set in range(sets):
			for w in range(ws):
				wpos = _set*ws + w

				dw = self.train._meand[wpos] / lines			
				s = self.S[wpos] = self.CONSTS['BETA0']*self.S[wpos] + (1-self.CONSTS['BETA0'])*dw
				v = self.V[wpos] = self.CONSTS['BETA1']*self.V[wpos] + (1-self.CONSTS['BETA1'])*dw*dw

				_s = s / (1 - self.CONSTS['BETA0']**self.echopes)
				_v = v / (1 - self.CONSTS['BETA1']**self.echopes)

				if type(self.CONSTS['ALPHA'] * _s * (_v + 1e-8)**(-.5)) is complex:
					breakpoint()

				self.train._weight[wpos] -= self.CONSTS['ALPHA'] * _s * (_v + 1e-8)**(-.5)