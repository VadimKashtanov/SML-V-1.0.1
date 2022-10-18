from kernel.py.optis import Score_Patern

class MEANSQUARED(Score_Patern):
	name = "MEAN SQUARED"

	description = '''Loss(want,get) := 0.5 * (want - get)^2
d(Loss(want,get))/d(get) = 0.5*d(want^2 - 2*want*get + get^2)/d(get)
						 = 0.5*(0 - 2*want + 2*get)
						 = want - get
Score of a set = sum(output for lines for outputs) / (lines * outputs)
'''

	CONSTS = {}

	def __init__(self, opti):
		self.opti_class = opti
		self.train = opti.train

	def __del__(self):
		pass

	def loss(self):
		train = self.train
		mdl = train.model
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		scores = [0 for s in range(sets)]

		for s in range(sets):
			for l in range(lines):
				for o in range(outputs):
					pos = l*sets*total + s*total + outstart + o

					#	(get - want)**2/2
					train._grad[pos] = .5*(train._var[pos] - data.output[l*outputs + o])**2
					scores[s] += .5*(train._var[pos] - data.output[l*outputs + o])**2
			scores[s] /= lines * outputs

		del self.opti_class.podium

		sorted_ = list(sorted(enumerate(scores), key=lambda x:x[1]))

		self.opti_class.podium = [_set for _set,score in sorted_]	#enumerate([0.1,0.2]) => [(0,0.1), (1,0.2)]   so (set,score)

		for rank,(_set,score) in enumerate(sorted_):
			self.opti_class.set_score[_set] = score
			self.opti_class.set_rank[_set] = rank

	def dloss(self):
		train = self.train
		mdl = train.model
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		for l in range(lines):
			for s in range(sets):
				for o in range(outputs):
					#get - want
					#so : get -= want, but for clarity we will leav it as it is
					train._grad[(l*sets*total) + (s*total) + (outstart + o)] = train._var[(l*sets*total) + (s*total) + (outstart + o)] - train.data.output[(l*outputs) + o]