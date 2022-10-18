import struct as st

class Opti_Class:
	def __init__(self, train, score=None, opti=None):
		self.train = train

		if score != None:
			self.score = score(self)
		else:
			self.score = None

		if opti != None:
			self.opti = opti(self)
		else:
			self.opti = None

		self.check()

		self.set_score = [0 for i in range(train.sets)]
		self.set_rank = [0 for i in range(train.sets)]

		self.podium = [0 for i in range(train.sets)]

	def __del__(self):
		del self.score
		del self.opti

		del self.set_score
		del self.set_rank
		del self.podium

	def check(self):
		if self.opti:
			self.opti.check()
		if self.score:
			self.score.check()

	###############################

	def bin_score(self):
		return st.pack('f'*len(self.set_score), *self.set_score)

	def bin_rank(self):
		return st.pack('I'*len(self.set_score), *self.set_score)

	def bin_podium(self):
		return st.pack('I'*len(self.set_score), *self.set_score)

	###############################

	def loss(self):				# -> compute scores and podium
		self.score.loss()

	def dloss(self):
		self.score.dloss()

	###############################

	def optimize(self):
		self.opti.opti()

class Opti_Patern:
	name = None
	description = None
	CONSTS = None

	MIN_TEST_ECHOPES = None

	def bins(self):
		bins = b''

		bins += st.pack('I', len(self.CONSTS.keys()))

		for k,v in self.CONSTS.items():
			bins += st.pack('I', len(k))
			bins += k.encode()
			bins += st.pack('I', len(str(v)))
			bins += str(v).encode()

		return bins

	def __del__(self):
		raise Exception("Not writed")

	def opti(self):
		raise Exception("Not writed")

	def check(self):
		assert self.name != None
		assert self.description != None
		assert self.CONSTS != None
		assert self.MIN_TEST_ECHOPES != None

class Score_Patern:
	name = None
	description = None
	CONSTS = None

	def bins(self):
		bins = b''

		bins += st.pack('I', len(self.CONSTS.keys()))

		for k,v in self.CONSTS.items():
			bins += st.pack('I', len(k))
			bins += k.encode()
			bins += st.pack('I', len(str(v)))
			bins += str(v).encode()

		return bins

	def __del__(self):
		raise Exception("Not writed")

	def loss(self):
		raise Exception("Not writed")

	def dloss(self):
		raise Exception("Not writed")

	def check(self):
		assert self.name != None
		assert self.description != None
		assert self.CONSTS != None