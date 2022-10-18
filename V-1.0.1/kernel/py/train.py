import struct as st

from kernel.py.etc import pseudo_randomf_minus1_1

class Train:
	def __init__(self, model, data, sets):
		self.model = model
		self.data = data
		self.sets = sets

		ws = model.weights
		total = model.total
		lines = data.lines
		locds = model.locds

		self._weight = [0 for _ in range(sets * ws)]
		self._var = [0 for _ in range(sets * lines * total)]
		self._locd = [0 for _ in range(sets * lines * locds)]
		self._grad = [0 for _ in range(sets * lines * total)]
		self._meand = [0 for _ in range(sets * ws)]

	def check(self):
		if 0 in self._weight:
			print("[train.py] Warrning 0 in train._weight")

	def __del__(self):
		del self._weight
		del self._var
		del self._locd
		del self._grad
		del self._meand

	####################

	def randomize(self, seed):
		ws = self.model.weights

		for s in range(self.sets):
			for w in range(ws):
				#	Il faut faire attention a ca, il faut que tout soit identique entre Cuda et Python
				wpos = s*ws + w
				self._weight[ wpos ] = pseudo_randomf_minus1_1(seed + wpos)

	####################

	def bin_w(self):
		return st.pack('f'*len(self._weight), *self._weight)

	def bin_v(self):
		return st.pack('f'*len(self._var), *self._var)

	def bin_l(self):
		return st.pack('f'*len(self._locd), *self._locd)

	def bin_g(self):
		return st.pack('f'*len(self._grad), *self._grad)

	def bin_m(self):
		return st.pack('f'*len(self._meand), *self._meand)

	def bins(self):
		return self.bin_w() + self.bin_v() + self.bin_l() + self.bin_g() + self.bin_m()
		
	#######################

	def set_inputs(self, batch=0):
		inputs = self.model.inputs
		lines = self.data.lines

		for i in range(inputs):
			for s in range(self.sets):
				for time in range(lines):
					self._var[time*self.sets*self.model.total + s*self.model.total + i] = self.data.input[batch*lines*inputs + time*inputs + i]

	def null_grad_meand(self):
		for i in range(len(self._grad)):
			self._grad[i] = 0
		for i in range(len(self._meand)):
			self._meand[i] = 0

	def prepare(self, batch):
		self.set_inputs(batch)
		self.null_grad_meand()

	########################

	def forward(self):
		for time in range(self.data.lines):
			for inst in self.model.insts:
				for _set in range(self.sets):
					inst.forward(
						self.sets, self.model.total, self.model.weights, self.model.locds, 
						_set, time,
						self._weight, self._var, self._locd)

	def backward(self):
		for time in list(range(self.data.lines))[::-1]:
			for inst in self.model.insts[::-1]:
				for _set in range(self.sets):
					inst.backward(
						self.sets, self.model.total, self.model.weights, self.model.locds, 
						_set, time,
						self._weight, self._var, self._locd, self._grad, self._meand)

	###########################################################################
	###########################################################################
	###########################################################################

	def print_weights(self):
		model = self.model
		ws = model.weights

		labels, poss = list(zip(*model.wsep))
		
		for s in range(self.sets):
			color = (93 if s % 2 else 96)

			print(f"\033[{color}m || \033[0m Set #{s}")
			for w in range(ws):
				if w in poss:
					print(f"\033[{color}m ||| --> \033[0m {labels[poss.index(w)]}")	
				print(f"\033[{color}m ||| {w}|\033[0m {self._weight[ws*s + w]}")

	def print_vars(self):
		model = self.model
		ws = model.weights

		labels, poss = list(zip(*model.vsep))

		for l in range(self.lines):
			color_l = (92 if l % 2 else 91)

			print(f"\033[{color_l}m || \033[0m Line #{l}")
			for s in range(self.sets):
				color_s = (93 if s % 2 else 96)

				print(f"\033[{color_l}m ||\033[{color_s}m||\033[0m Set #{s}")
				for i in range(model.total):
					if i in poss:
						print(f"\033[{color_l}m ||\033[{color_s}m|| --> \033[0m {labels[poss.index(i)]}")	
					print(f"\033[{color_l}m ||\033[{color_s}m|| {i}| \033[0m {self._var[l*model.total*self.sets + s*model.total + i]}")

	def print_locds(self):
		model = self.model
		ws = model.weights

		labels, poss = list(zip(*model.lsep))

		for l in range(self.lines):
			color_l = (92 if l % 2 else 91)

			print(f"\033[{color_l}m || \033[0m Line #{l}")
			for s in range(self.sets):
				color_s = (93 if s % 2 else 96)

				print(f"\033[{color_l}m ||\033[{color_s}m||\033[0m Set #{s}")
				for i in range(model.locds):
					if i in poss:
						print(f"\033[{color_l}m ||\033[{color_s}m|| --> \033[0m {labels[poss.index(i)]}")	
					print(f"\033[{color_l}m ||\033[{color_s}m|| {i}| \033[0m {self._locd[l*model.locds*self.sets + s*model.locds + i]}")


	def print_grads(self):
		model = self.model
		ws = model.weights

		labels, poss = list(zip(*model.vsep))

		for l in range(self.lines):
			color_l = (92 if l % 2 else 91)

			print(f"\033[{color_l}m || \033[0m Line #{l}")
			for s in range(self.sets):
				color_s = (93 if s % 2 else 96)

				print(f"\033[{color_l}m ||\033[{color_s}m||\033[0m Set #{s}")
				for i in range(model.total):
					if i in poss:
						print(f"\033[{color_l}m ||\033[{color_s}m|| --> \033[0m {labels[poss.index(i)]}")	
					print(f"\033[{color_l}m ||\033[{color_s}m|| {i}| \033[0m {self._grad[l*model.total*self.sets + s*model.total + i]}")


	def print_meands(self):
		model = self.model
		ws = model.weights

		labels, poss = list(zip(*model.wsep))
		
		for s in range(self.sets):
			color = (93 if s % 2 else 96)

			print(f"\033[{color}m || \033[0m Set #{s}")
			for w in range(ws):
				if w in poss:
					print(f"\033[{color}m ||| --> \033[0m {labels[poss.index(w)]}")	
				print(f"\033[{color}m ||| {w}|\033[0m {self._meand[ws*s + w]}")