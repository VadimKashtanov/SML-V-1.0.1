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

class DOT1D(BuildFromRequired):

	name = "DOT1D"

	params_names = ['Ax','Yx', 'activ', 'input_start','ystart','wstart','locdstart', 'drop_rate']

	################################ Kernel Functions ##########################################
	def check(self):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		assert activ < len(activate) and Ax > 0 and Yx > 0 and 100 >= drop_rate >= 0 and all(i>=0 and int(i)==i for i in self.params)

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params

		for y in range(Yx):
			_sum = 0
			
			for i in range(Ax):
				_sum += var[l*total + input_start + i] * w[wstart + i*Yx + y]
			_sum += w[wstart + Ax*Yx + y]

			var[l*total + ystart + y] = activate[activ](_sum)

	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params

		for y in range(Yx):
			_sum = 0
			for i in range(Ax):
				_sum += var[sets*total*line + _set*total + input_start + i] * w[ws*_set + wstart + i*Yx + y]
			_sum += w[ws*_set + wstart + Ax*Yx + y]

			locd[sets*line*locds + _set*locds + locdstart + y] = localderiv[activ](_sum)
			var[sets*total*line + _set*total + ystart + y] = activate[activ](_sum)

	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params

		for y in range(Yx):
			dlds = locd[sets*line*locds + _set*locds + locdstart + y] * grad[sets*total*line + _set*total + ystart + y]

			meand[ws*_set + wstart + Ax*Yx + y] += dlds

			for i in range(Ax):
				wpos = ws*_set + wstart + i*Yx + y
				vpos = sets*total*line + _set*total + input_start + i
				grad[vpos] += dlds * w[wpos]
				meand[wpos] += dlds * var[vpos]

	####################### Spetial functions for applications ##########################

	#### Build Stack Model  (Applications : "stack_model.py", )

	def buildstackmodel_vars(self):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return Yx

	def buildstackmodel_weights(self):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return Ax*Yx + Yx

	def buildstackmodel_locds(self):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return Yx

	#### Labels Stack Model  (Applications : "stack_model.py", )

	def labelstackmodel_vars(self, _id, stack_start):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return [(f'{_id}.Y [dot1d]',stack_start)]

	def labelstackmodel_weights(self, _id, stack_start):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return [(f'{_id}.W [dot1d]',stack_start), (f'{_id}.B [dot1d]',stack_start + Ax*Yx)]

	def labelstackmodel_locds(self, _id, stack_start):
		Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return [(f'{_id}.Y [dot1d]',stack_start)]

	### Setput Params Stack Model  (Applications : "stack_model.py", )

	requiredforsetupparams = "Ax", "Yx", "activ", "drop_rate"		#build vars,weights and locd have to ask only for thoses params

	requiredposition = 1,1,1,0,0,0,0,1 #1,1,1 == Ax,Yx,activ

	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		Ax, Yx, activ, drop_rate = required
		return Ax, Yx, activ, istart, ystart, wstart, lstart, drop_rate

	### Check Params Input output

	def need_inputs(self, required):
		Ax, Yx, activ, drop_rate = required
		return Ax

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		Ax, Yx, activ, drop_rate = required				#que le input pour cet instruction
		assert last_vars == self.need_inputs(required)