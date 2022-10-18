from math import exp, tanh

activate = [
	lambda x: 1 / (1 + exp(-x)),
	tanh,
	lambda x: exp(-x*x),
	lambda x: x * (x >= 0),
	lambda x: x,
]

localderiv = [
	lambda x: activate[0](x) * (1 - activate[0](x)),
	lambda x: 1 - tanh(x)**2,
	lambda x: -2*x*activate[2](x),
	lambda x: (x >= 0),
	lambda x: 1,
]

from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired

class ACTIVATION(BuildFromRequired):

	name = "ACTIVATION"

	params_names = ['_len', 'activ', 'istart','ystart','wstart','locdstart']

	################################ Kernel Functions ##########################################
	def check(self):
		_len, activ, istart, ystart, wstart, locdstart = self.params
		assert activ < len(activate) and _len > 0 and all(i>=0 and int(i)==i for i in self.params)

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		_len, activ, istart, ystart, wstart, locdstart = self.params

		for i in range(_len):
			var[l*total + ystart + i] = activate[activ](var[l*total + istart + i])

	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):
		
		_len, activ, istart, ystart, wstart, locdstart = self.params

		for i in range(_len):
			locd[sets*line*locds + _set*locds + locdstart + i] = localderiv[activ](var[sets*total*line + _set*total + istart + i])
			var[sets*total*line + _set*total + ystart + i] = activate[activ](var[sets*total*line + _set*total + istart + i])

	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		_len, activ, istart, ystart, wstart, locdstart = self.params

		for i in range(_len):
			dlds = grad[sets*total*line + _set*total + ystart + i] * locd[sets*line*locds + _set*locds + locdstart + i]
			
			grad[sets*total*line + _set*total + istart + i] += dlds

	####################### Spetial functions for applications ##########################

	#### Build Stack Model  (Applications : "stack_model.py", )

	def buildstackmodel_vars(self):
		_len, activ, istart, ystart, wstart, locdstart = self.params
		return _len

	def buildstackmodel_weights(self):
		_len, activ, istart, ystart, wstart, locdstart = self.params
		return 0

	def buildstackmodel_locds(self):
		_len, activ, istart, ystart, wstart, locdstart = self.params
		return _len

	#### Labels Stack Model  (Applications : "stack_model.py", )

	def labelstackmodel_vars(self, _id, stack_start):
		_len, activ, istart, ystart, wstart, locdstart = self.params
		return [(f'{_id}.Y [activation]',stack_start)]

	def labelstackmodel_weights(self, _id, stack_start):
		_len, activ, istart, ystart, wstart, locdstart = self.params
		return []

	def labelstackmodel_locds(self, _id, stack_start):
		_len, activ, istart, ystart, wstart, locdstart = self.params
		return [(f'{_id}.Y [activation]',stack_start)]

	### Setput Params Stack Model  (Applications : "stack_model.py", )

	requiredforsetupparams = "_len", "activ"		#build vars,weights and locd have to ask only for thoses params

	requiredposition = 1,1,0,0,0,0 #1,1,1 == Ax,Yx,activ

	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		_len, activ = required
		return _len, activ, istart, ystart, wstart, lstart

	### Check Params Input output

	def need_inputs(self, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		_len, activ = required
		return _len

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		_len, activ = required		#que le input pour cet instruction
		assert last_vars == self.need_inputs(required)