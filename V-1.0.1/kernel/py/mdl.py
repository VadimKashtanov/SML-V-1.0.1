import struct as st

from kernel.py.inst import Inst
from random import random

class Mdl:
	def __init__(self, insts, inputs, outputs, _vars, w, locds, vsep, wsep, lsep):
		#	Check if insts are all initialled

		for inst in insts:
			inst.check()

		self.insts = insts

		self.inputs = inputs
		self.outputs = outputs

		#self.vars = _vars
		self.weights = len(w)
		self.w = w

		self.locds = locds

		self.total = _vars + self.inputs

		#	Separators
		self.vsep = vsep
		self.wsep = wsep
		self.lsep = lsep

	def check(self):
		for inst in self.insts:
			inst.check()

		assert self.vsep != None
		assert self.wsep != None
		assert self.lsep != None

	def print_insts(self):
		for i,inst in enumerate(self.insts):
			print(f"{i}| {inst.name}   " + ' '.join(inst.params_name[i] + '=' + str(inst.params[i]) for i in range(len(inst.params))))

	def print_weights(self):
		ws = self.weights

		labels, poss = list(zip(*self.wsep))
		
		for w in range(ws):
			if w in poss:
				print(f" {labels[poss.index(w)]}")	
			print(f"{w}|  {self.w[w]}")

	def bins(self):
		bins = st.pack('I', len(self.insts))

		for inst in self.insts:
			bins += st.pack('I', inst.ID)
			bins += st.pack('I'*len(inst.params), *inst.params)

		bins += st.pack('IIIII', self.inputs, self.outputs, (_vars:=(self.total-self.inputs)), self.weights, self.locds)

		bins += st.pack('f'*len(self.w), *self.w)

		#	Separators
		for sep in self.vsep,self.wsep,self.lsep:
			bins += st.pack('I', len(sep))
			for lbl,pos in sep:
				bins += st.pack('I', len(lbl))
				bins += lbl.encode()
				bins += st.pack('I', pos)

		return bins