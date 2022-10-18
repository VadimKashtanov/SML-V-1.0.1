#langue ou tu ecrit le cuda dans un petit fichier, les constants et un peut de python-like et ca te genere tout un package.
#poincarelang (pcl) .pcl
#ca te genere le C/Cuda, le Python et autres langues si il le faut

from kernel.py.optis import Opti_Patern

class SGD(Opti_Patern):
	name = "SGD"

	description = '''w -= alpha * grad(w)'''

	CONSTS = {
		'ALPHA' : 1e-5
	}

	MIN_TEST_ECHOPES = 1

	def __init__(self, opti):
		self.train = opti.train

	def __del__(self):
		pass

	def opti(self):
		mdl = self.train.model
		sets = self.train.sets
		ws = mdl.weights
		lines = self.train.data.lines

		for s in range(sets):
			for w in range(ws):
				self.train._weight[s*ws + w] -= self.CONSTS['ALPHA'] * self.train._meand[s*ws + w] / lines