from math import exp

from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired

class SOFTMAX(BuildFromRequired):
	name='SOFTMAX'
	params_names=['_len', 'input_start', 'ystart']

	def check(self):
		assert all(i>=0 and int(i)==i for i in self.params)
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
		_len, input_start, ystart = self.params
		_sum = 0
		for i in range(_len):
			var[l*total + ystart + i] = exp(var[l*total + input_start + i])
			_sum += var[l*total + ystart + i]
		for i in range(_len):
			var[l*total + ystart + i] /= _sum
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		_len, input_start, ystart = self.params
		
		_sum = 0
		for i in range(_len):
			var[line*sets*total + _set*total + ystart + i] = exp(var[line*sets*total + _set*total + input_start + i])
			_sum += var[line*sets*total + _set*total + ystart + i]
		for i in range(_len):
			var[line*sets*total + _set*total + ystart + i] /= _sum
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		_len, input_start, ystart = self.params
		
		for i in range(_len):
			err = grad[line*sets*total + _set*total + ystart + i]
			
			for j in range(_len):
				yi = var[line*sets*total + total*_set + ystart + i]
				yj = var[line*sets*total + total*_set + ystart + j]

				if i == j:
					grad[line*sets*total + _set*total + input_start + j] += err * yi * (1 - yi)
				else:
					grad[line*sets*total + _set*total + input_start + j] += - err * yi * yj

	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		_len, input_start, ystart = self.params
		return _len
	
	def buildstackmodel_weights(self):
		_len, input_start, ystart = self.params
		return 0
	
	def buildstackmodel_locds(self):
		_len, input_start, ystart = self.params
		return 0
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		_len, input_start, ystart = self.params
		return [(f'{_id}.Y [softmax]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		_len, input_start, ystart = self.params
		return []
	
	def labelstackmodel_locds(self,_id, stack_start):
		_len, input_start, ystart = self.params
		return []
	
	### Setput Params Stack Model
	
	requiredforsetupparams = "_len",
	
	requiredposition = 1,0,0
	
	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		_len, = required
		return _len,istart,ystart

	### Check Params Input output

	def need_inputs(self, required):
		_len, = required
		return _len

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		_len, = required 								#que le input pour cet instruction
		assert last_vars == self.need_inputs(required)