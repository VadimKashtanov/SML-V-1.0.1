from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired

'''
size = 3
   [1, 2, 3]
+  [6, 1, 5]
+  [1, 1, 0]	items = 4
+  [9, 1, 0]
=  [17,5, 8]

'''

class SUM(BuildFromRequired):
	name='SUM'
	params_names=['size', 'items', 'istart', 'ystart']

	def check(self):
		assert all(i>=0 and int(i)==i for i in self.params)
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
		
		size, items, istart, ystart = self.params
		
		for elm in range(size):
			_sum = 0

			for item in range(items):
				_sum += var[l*total + istart + item*size + elm]

			var[l*total + ystart + elm] += _sum
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		
		size, items, istart, ystart = self.params
		
		for elm in range(size):
			_sum = 0

			for item in range(items):
				_sum += var[line*sets*total + _set*total + istart + item*size + elm]

			var[line*sets*total + _set*total + ystart + elm] += _sum
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		size, items, istart, ystart = self.params
		
		for elm in range(size):
			dlds = grad[line*total + ystart + elm]

			for item in range(items):
				grad[line*total + istart + item*size + elm] += dlds

	####################### Spetial functions for applications ##########################	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		size, items, istart, ystart = self.params
		return size
	
	def buildstackmodel_weights(self):
		size, items, istart, ystart = self.params
		return 0
	
	def buildstackmodel_locds(self):
		size, items, istart, ystart = self.params
		return 0
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		size, items, istart, ystart = self.params
		return [(f'{_id}.Y [sum]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		size, items, istart, ystart = self.params
		return []
	
	def labelstackmodel_locds(self,_id, stack_start):
		_len, input_start, ystart = self.params
		return []
	
	### Setput Params Stack Model
	
	requiredforsetupparams = "size", "items"
	
	requiredposition = 1,1,0,0
	
	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		size, items, = required
		return size, items, istart, ystart

	### Check Params Input output

	def need_inputs(self, required):
		size, items = required
		return size * items

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		size, items, = required 								#que le input pour cet instruction
		assert last_vars == self.need_inputs(required)