class Inst:
	name = None
	ID = None

	params_names = None

	def __init__(self, preset=None):
		self.params_dict = {}
		self.params = []

		for i,param in enumerate(self.params_names):
			if preset != None:
				if type(preset[i]) is int and preset[i] >= 0:
					self.params_dict[param] = preset[i]
					self.params += [preset[i]]
				else:
					self.params_dict[param] = None
					self.params += [None]
			else:
				self.params_dict[param] = None
				self.params += [None]
		
		#p:(preset[p] if type(preset[p]) is int and preset[p] >= 0 else None) for i,p in enumerate(self.params_names)}

	def __getitem__(self, key):
		return self.params[key]#self.params_names[key]]

	def __setitem__(self, key, val):
		if not type(val) is int:
			raise Exception(f"Can't assign none positiv int to a parameter because kernel use `uint`. ({val} is not int type)")
		
		self.params[key] = val

	def check(self):
		for k,v in self.params:
			assert v != None
		assert self.ID != None
		assert self.name != None
		assert self.params_names != None

	def mdl(self, total, line, var, w):
		assert 0

	def forward(self, sets, total, ws, locds, _set, line, w, var, locd):
		assert 0

	def backward(self, sets, total, ws, locds, _set, line, w, var, locd, grad, meand):
		assert 0