from math import exp

'''
y will be in [0;1]

		      [p0,p1,p2]
[a0,a1,a2] -> [c0,c1,c2]
[b0,b1,b2] -> [d0,d1,d2]

c0 = exp(-(a0 + p0)^2)
d0 = exp(-(b0 + p0)^2)

locd_c0 = -2*(a0+p0)*c0
locd_d2 = -2*(b2+p2)*d2

'''

'''	Var
X*Y
'''
'''	Weights
X
'''
'''	Lods
X*Y
'''

#Params : [X,Y, istart,ystart,wstart,lstart]

from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired

class GAUSSFILTRE2D(BuildFromRequired):
	name='GAUSSFILTRE2D'
	params_names=['X', 'Y', 'istart', 'ystart', 'wstart', 'lstart']

	def check(self):
		X,Y, istart,ystart,wstart,lstart = self.params
		assert all(i >= 0 for i in self.params) and X > 0 and Y > 0
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
	
		X,Y, istart,ystart,wstart,lstart = self.params
	
		inp = l*total + istart
		out = l*total + ystart
		ppos = wstart
	
		for y in range(Y):
			for x in range(X):
				a = var[inp + y*X + x]
				p = w[ppos + x]
				var[out + y*X + x] = exp(-(a+p)**2)
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		
		X,Y, istart,ystart,wstart,lstart = self.params
	
		inp = line*sets*total + _set*total + istart
		out = line*sets*total + _set*total + ystart
		ppos = _set*ws + wstart
	
		for y in range(Y):
			for x in range(X):
				a = var[inp + y*X + x]
				p = w[ppos + x]
				var[out + y*X + x] = exp(-(a+p)**2)
				locd[line*sets*locds + _set*locds + lstart + y*X + x] = -2*(a+p)*exp(-(a+p)**2)
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
	
		X,Y, istart,ystart,wstart,lstart = self.params
	
		inp = line*sets*total + _set*total + istart
		out = line*sets*total + _set*total + ystart
		ppos = _set*ws + wstart
	
		for y in range(Y):
			for x in range(X):
				pos = y*X + x
				dy = grad[out + pos]
	
				dw = locd[line*sets*locds + _set*locds + lstart + y*X + x] * dy
	
				grad[inp + pos] += dw
				meand[ppos + x] += dw
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		X,Y, istart,ystart,wstart,lstart = self.params
		return X*Y
	
	def buildstackmodel_weights(self):
		X,Y, istart,ystart,wstart,lstart = self.params
		return X
	
	def buildstackmodel_locds(self):
		X,Y, istart,ystart,wstart,lstart = self.params
		return X*Y
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		X,Y, istart,ystart,wstart,lstart = self.params
		return [(f'{_id}.Y [gaussfiltre2d]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		X,Y, istart,ystart,wstart,lstart = self.params
		return [(f'{_id}.P [gaussfiltre2d]',stack_start)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		X,Y, istart,ystart,wstart,lstart = self.params
		return [(f'{_id}.Y [gaussfiltre2d]',stack_start)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_gaussfiltre2d = "X", "Y"
	
	requiredposition_gaussfiltre2d = 1,1,0,0,0,0 #1,1,1 == Ax,Yx,activ
	
	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		X,Y = required
		return X,Y, istart,ystart,wstart,lstart

	### Check Params Input output

	def need_inputs(self, required):
		X,Y = required
		return X*Y

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		X,Y = required
		assert last_vars == self.need_inputs(required)