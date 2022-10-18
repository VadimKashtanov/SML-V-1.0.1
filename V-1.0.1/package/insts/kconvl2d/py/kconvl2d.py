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

'''
		On voit ici que les dimentions des tensor sont X,Y
		La raison est le fait que ça soit de dimention 2D.
		Plus tard je pourrais implementer avec des tenseurs.

		Pour l'instant je fais un truc simple. Et les models
		de genetique seront construits avec ces instructions simples
		(donc de dimention finie)
'''

class KCONVL2D(BuildFromRequired):

	name = "KCONVL2D"

	params_names = ['Ax','Ay', 'Kx', 'Ky', 'n0', 'n1', 'strideX', 'strideY', 'paddingX', 'paddingY', 'activ', 'input_start','ystart','wstart','locdstart', 'drop_rate']

	################################ Kernel Functions ##########################################
	def check(self):
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		assert strideX > 0 and strideY > 0 and Kx % 2 != 0 and Ky % 2 != 0 and activ < len(activate) and Ax > 0 and Ay > 0 and 100 >= drop_rate >= 0 and all(i>=0 and int(i)==i for i in self.params) and n0 > 0 and n1 > 0
		assert (Ax - 2*paddingX) % strideX == 0
		assert (Ay - 2*paddingY) % strideY == 0

	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):

		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, istart, ystart, wstart, locdstart, drop_rate = self.params

		ker_radiusX = int((Kx-1)/2)
		ker_radiusY = int((Ky-1)/2)

		Yx = int((Ax - 2*paddingX) / strideX)
		Yy = int((Ay - 2*paddingY) / strideY)

		for _n1 in range(n1):

			for x in range(paddingX, Ax-paddingX, strideX):
				for y in range(paddingY, Ay-paddingY, strideY):

					_sum = 0

					for _n0 in range(n0):
						for ker_x in range(-ker_radiusX, ker_radiusX+1):	#[-1,0,1]
							for ker_y in range(-ker_radiusY, ker_radiusY+1):	#[-1,0,1]
								if Ay > (y + ker_y) >= 0 and Ax > (x + ker_x) >= 0:
									_pixelpos = l*total + istart + _n0*Ax*Ay + (y+ker_y)*Ax + (x+ker_x)
									_kernelpos = wstart + _n1*Kx*Ky*n0 + _n0*Kx*Ky + ((ker_y+ker_radiusY)*Kx + ker_x+ker_radiusX)

									_sum += var[_pixelpos] * w[_kernelpos]

					_pixelpos = int(_n1*Yx*Yy + ((y-paddingY)/strideY)*Yx + ((x-paddingX)/strideX))
					_sum += w[wstart + n1*n0*Kx*Ky + _pixelpos]
					var[l*total + ystart + _pixelpos] = activate[activ](_sum)

		#breakpoint()

	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float]):
		
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, istart, ystart, wstart, locdstart, drop_rate = self.params

		ker_radiusX = int((Kx-1)/2)
		ker_radiusY = int((Ky-1)/2)

		Yx = int((Ax - 2*paddingX) / strideX)
		Yy = int((Ay - 2*paddingY) / strideY)

		for _n1 in range(n1):

			for x in range(paddingX, Ax-paddingX, strideX):
				for y in range(paddingY, Ay-paddingY, strideY):

					_sum = 0

					for _n0 in range(n0):
						for ker_x in range(-ker_radiusX, ker_radiusX+1):	#[-1,0,1]
							for ker_y in range(-ker_radiusY, ker_radiusY+1):	#[-1,0,1]
								if Ay > (y + ker_y) >= 0 and Ax > (x + ker_x) >= 0:
									_pixelpos = line*sets*total + _set*total + istart + _n0*Ax*Ay + (y+ker_y)*Ax + (x+ker_x)
									_kernelpos = _set*ws + wstart + _n1*Kx*Ky*n0 + _n0*Kx*Ky + ((ker_y+ker_radiusY)*Kx + ker_x+ker_radiusX)

									_sum += var[_pixelpos] * w[_kernelpos]

					_pixelpos = int(_n1*Yx*Yy + ((y-paddingY)/strideY)*Yx + ((x-paddingX)/strideX))
					_sum += w[_set*ws + wstart + n1*n0*Kx*Ky + _pixelpos]

					var[line*sets*total + _set*total + ystart + _pixelpos] = activate[activ](_sum)
					locd[line*sets*locds + _set*locds + locdstart + _pixelpos] = localderiv[activ](_sum)

	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int,
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, istart, ystart, wstart, locdstart, drop_rate = self.params

		ker_radiusX = int((Kx-1)/2)
		ker_radiusY = int((Ky-1)/2)

		Yx = int((Ax - 2*paddingX) / strideX)
		Yy = int((Ay - 2*paddingY) / strideY)

		for _n1 in range(n1):

			for x in range(paddingX, Ax-paddingX, strideX):
				for y in range(paddingY, Ay-paddingY, strideY):

					_pixelpos = int(_n1*Yx*Yy + ((y-paddingY)/strideY)*Yx + ((x-paddingX)/strideX))

					#dLoss/dsum
					dlds = grad[line*sets*total + _set*total + ystart + _pixelpos] * locd[line*sets*locds + _set*locds + locdstart + _pixelpos]

					#.B meand
					meand[_set*ws + wstart + n1*n0*Kx*Ky + _pixelpos] += dlds

					_sum = 0

					for _n0 in range(n0):
						for ker_x in range(-ker_radiusX, ker_radiusX+1):	#[-1,0,1]
							for ker_y in range(-ker_radiusY, ker_radiusY+1):	#[-1,0,1]
								if Ay > (y + ker_y) >= 0 and Ax > (x + ker_x) >= 0:
									_pixelpos = line*sets*total + _set*total + istart + _n0*Ax*Ay + (y+ker_y)*Ax + (x+ker_x)
									_kernelpos = _set*ws + wstart + _n1*Kx*Ky*n0 + _n0*Kx*Ky + ((ker_y+ker_radiusY)*Kx + ker_x+ker_radiusX)

									grad[_pixelpos] += dlds * w[_kernelpos]
									meand[_kernelpos] += dlds * var[_pixelpos]
									
	####################### Spetial functions for applications ##########################

	#### Build Stack Model  (Applications : "stack_model.py", )

	def buildstackmodel_vars(self):
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return int(((Ax-2*paddingX)/strideX)*((Ay-2*paddingY)/strideY) * n1)

	def buildstackmodel_weights(self):
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return int(((Ax-2*paddingX)/strideX)*((Ay-2*paddingY)/strideY) * n1 + n0*n1*Kx*Ky)

	def buildstackmodel_locds(self):
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return int(((Ax-2*paddingX)/strideX)*((Ay-2*paddingY)/strideY) * n1)

	#### Labels Stack Model  (Applications : "stack_model.py", )

	def labelstackmodel_vars(self, _id, stack_start):
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return [(f'layer {_id} kconvl2d.Y',stack_start)]

	def labelstackmodel_weights(self, _id, stack_start):
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return [(f'layer {_id} kconvl2d.W',stack_start), (f'layer {_id} kconvl2d.B',stack_start + n0*n1*Kx*Ky)]

	def labelstackmodel_locds(self, _id, stack_start):
		Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, input_start, ystart, wstart, locdstart, drop_rate = self.params
		return [(f'layer {_id} kconvl2d.Y',stack_start)]

	### Setput Params Stack Model  (Applications : "stack_model.py", )

	requiredforsetupparams = "Ax", "Ay", "Kx", "Ky", "n0", "n1", "strideX", "strideY", "paddingX", "paddingY", "activ", "drop_rate"		#build vars,weights and locd have to ask only for thoses params

	requiredposition = 1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1

	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		Ax, Ay, Kx, Ky, n0, n1, strideX, strideY, paddingX, paddingY, activ, drop_rate = required
		return Ax, Ay, Kx, Ky, n0,n1, strideX, strideY, paddingX, paddingY, activ, istart, ystart, wstart, lstart, drop_rate

	### Check Params Input output

	def need_inputs(self, required):
		Ax, Ay, Kx, Ky, n0, n1, strideX, strideY, paddingX, paddingY, activ, drop_rate = required
		return Ax*Ay*n0

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		Ax, Ay, Kx, Ky, n0, n1, strideX, strideY, paddingX, paddingY, activ, drop_rate = required		#que le input pour cet instruction
		assert last_vars == self.need_inputs(required)