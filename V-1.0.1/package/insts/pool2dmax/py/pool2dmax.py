from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired

class POOL2DMAX(BuildFromRequired):
	name = "POOL2DMAX"

	params_names = ['Ax','Ay', 'Xpool', 'Ypool', 'input_start','ystart','locdstart']

	################################ Kernel Functions ##########################################
	def check(self):
		Ax, Ay, Xpool, Ypool, input_start, ystart, locdstart = self.params
		assert (Ax % Xpool == 0 and Ay % Ypool == 0) and Ax > 0 and Ay > 0 and all(i>=0 and int(i)==i for i in self.params)

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		Ax, Ay, Xpool, Ypool, istart, ystart, locdstart = self.params
		
		Yx = int(Ax / Xpool)
		Yy = int(Ay / Ypool)

		for y in range(Yy):
			for x in range(Yx):
				_max = max(
					var[l*total + istart + (y*Ypool + _y)*Ax + (x*Xpool + _x)] 
						for _y in range(Ypool) 
							for _x in range(Xpool)
				)

				var[l*total + ystart + y*Yx + x] = _max

	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):
		
		Ax, Ay, Xpool, Ypool, istart, ystart, locdstart = self.params

		Yx = int(Ax / Xpool)
		Yy = int(Ay / Ypool)

		for y in range(Yy):
			for x in range(Yx):
				pixels = [
					var[line*sets*total + _set*total + istart + (y*Ypool + _y)*Ax + (x*Xpool + _x)] 
						for _y in range(Ypool)
							for _x in range(Xpool)
				]

				var[line*sets*total +  _set*total + ystart + y*Yx + x] = max(pixels)
				locd[line*sets*locds + _set*locds + locdstart + y*Yx + x] = float(pixels.index(max(pixels)))	#x = locd%Xpool, y = (locd-x)/Xpool

	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		Ax, Ay, Xpool, Ypool, istart, ystart, locdstart = self.params
		
		Yx = int(Ax / Xpool)
		Yy = int(Ay / Ypool)

		for y in range(Yy):
			for x in range(Yx):
				dldpoolmax = grad[line*sets*total + _set*total + ystart + y*Yx + x]

				index = locd_val = int(locd[line*sets*locds + _set*locds + locdstart + y*Yx + x])

				_x = index %Xpool
				if (index - _x) % Xpool != 0:
					raise Exception("(index-_x) have to be divised by Xpool")
				_y = int((index-_x)/Xpool)

				grad[line*sets*total + _set*total + istart + (y*Ypool + _y)*Ax + (x*Xpool + _x)] += dldpoolmax

	####################### Spetial functions for applications ##########################

	#### Build Stack Model  (Applications : "stack_model.py", )

	def buildstackmodel_vars(self):
		Ax, Ay, Xpool, Ypool, input_start, ystart, locdstart = self.params
		return int(Ax/Xpool * Ay/Ypool)

	def buildstackmodel_weights(self):
		Ax, Ay, Xpool, Ypool, input_start, ystart, locdstart = self.params
		return 0

	def buildstackmodel_locds(self):
		Ax, Ay, Xpool, Ypool, input_start, ystart, locdstart = self.params
		return int(Ax/Xpool * Ay/Ypool)

	#### Labels Stack Model  (Applications : "stack_model.py", )

	def labelstackmodel_vars(self, _id, stack_start):
		Ax, Ay, Xpool, Ypool, input_start, ystart, locdstart = self.params
		return [(f'layer {_id} pool2dmax.Y',stack_start)]

	def labelstackmodel_weights(self, _id, stack_start):
		Ax, Ay, Xpool, Ypool, input_start, ystart, locdstart = self.params
		return []

	def labelstackmodel_locds(self, _id, stack_start):
		Ax, Ay, Xpool, Ypool, input_start, ystart, locdstart = self.params
		return [(f'layer {_id} pool2dmax.Y',stack_start)]

	### Setput Params Stack Model  (Applications : "stack_model.py", )

	requiredforsetupparams = "Ax", "Ay", "Xpool", "Ypool"

	requiredposition = 1,1,1,1,0,0,0

	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		Ax, Ay, Xpool, Ypool = required
		return Ax, Ay, Xpool, Ypool, istart, ystart, lstart

	### Check Params Input output

	def need_inputs(self, required):
		Ax, Ay, Xpool, Ypool = required
		return Ax*Ay

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		Ax, Ay, Xpool, Ypool = required		#que le input pour cet instruction
		assert last_vars == self.need_inputs(required)