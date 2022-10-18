from math import exp, tanh

#lstm2d : [Ax,Ay] -> [Bx,Ay]

'''
fn = f(x@Wn + h[-1]@Un + bn)
 (				  =======				)
 (				  |     |				)
 (				  |		|				)
 (				  |	.W	|				)
 (				  |		|				)
 (				  |		|				)
 (				  =======				)
 (	============= =======				)
 (	|	.input	| | 	|	input@Wn    ) = f(input@Wn + h[-1]@Un + bn)
 (	============= =======				)
f(					 +					)
 (				  =======				)
 (				  |	.U  |				)
 (				  |	    |				)
 (				  =======				)
 (		  ======= =======				)
 (		  |h[-1]| |		|  h[-1]@Un 	)
 (		  ======= =======				)
 (					 + 					)
 (				  =======				)
 (				  |	.Bn	|				)
 (				  =======				)

x:[Ax,Ay]

f0 = f(x@Wf0 + h[-1]@Uf0 + Bf0)
f1 = f(x@Wf1 + h[-1]@Uf1 + Bf1)
f2 = f(x@Wf2 + h[-1]@Uf2 + Bf2)
g0 = g(x@Wg3 + h[-1]@Ug0 + Bg0)
e = f0 * e[-1] + f1 * g0
h = f2 * e

[1,3]*[4,5] = [1*4,3*5] c'est pas le produit cartesien mais une multiplication d'arrays (produit Hadamard)

h:[Bx:Ay]

f(x) = 1 / (1 + exp(-x))
g(x) = tanh(x)

Inputs = Ax*Ay
Outputs = (Bx*Ay)*2 (store e and h)

W:[Bx,Ax]
U:[Bx,Bx]
B:[Bx,Ay]
'''

'''	Var struct
e:[Bx,Ay]
h:[Bx,Ay]
'''

'''	Weight struct
Wf0:[Bx,Ax]
Uf0:[Bx,Bx]
Bf0:[Bx,Ay]

Wf1:[Bx,Ax]
Uf1:[Bx,Bx]
Bf1:[Bx,Ay]

Wf2:[Bx,Ax]
Uf2:[Bx,Bx]
Bf2:[Bx,Ay]

Wg0:[Bx,Ax]
Ug0:[Bx,Bx]
Bg0:[Bx,Ay]
'''

'''	Locd struct
f0:[Bx,Ay]
f1:[Bx,Ay]
f2:[Bx,Ay]
g0:[Bx,Ay]
'''

#	Params : [Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate]

from kernel.py.inst import Inst
from package.insts.build_from_required import BuildFromRequired

class LSTM2D(BuildFromRequired):
	name='LSTM2D'
	params_names=['Ax', 'Ay', 'Bx', 'istart', 'ystart', 'wstart', 'locdstart', 'drate']

	def check(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
		assert all(i >= 0 for i in self.params) and Ax > 0 and Ay > 0 and Bx > 0 and drate >= 0 and drate < 100
	
	def mdl(self,
		total:int, time:int,
		var:[float], w:[float]):
		
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params

		_W = Bx*Ax
		_U = Bx*Bx
		_B = Bx*Ay
	
		lineW = _W + _U + _B
	
		for y in range(Ay):
			for x in range(Bx):
				#### f0,f1,f2 = f(x@W + h[-1]@W + b)
				f0,f1,f2,g0 = 0, 0, 0, 0
				####
				# f0 is f0<x,y>
				####
	
				#### .W
				for k in range(Ax):
					vpos = total*time + istart + y*Ax + k
					
					f0 += var[vpos] * w[wstart + 0*lineW + (k*Bx + y)]
					f1 += var[vpos] * w[wstart + 1*lineW + (k*Bx + y)]
					f2 += var[vpos] * w[wstart + 2*lineW + (k*Bx + y)]
					g0 += var[vpos] * w[wstart + 3*lineW + (k*Bx + y)]
	
				#### .U
				if time > 0:
					for k in range(Bx):
						#								   (e)		(h[y][x])
						vpos = total*(time-1) + ystart + (Bx*Ay) + (y*Bx + k)

						f0 += var[vpos] * w[wstart + 0*lineW + _W + k*Bx + y]
						f1 += var[vpos] * w[wstart + 1*lineW + _W + k*Bx + y]
						f2 += var[vpos] * w[wstart + 2*lineW + _W + k*Bx + y]
						g0 += var[vpos] * w[wstart + 3*lineW + _W + k*Bx + y]
	
				#### .B
				f0 += w[wstart + 0*lineW + _W + _U + (y*Bx + x)]
				f1 += w[wstart + 1*lineW + _W + _U + (y*Bx + x)]
				f2 += w[wstart + 2*lineW + _W + _U + (y*Bx + x)]
				g0 += w[wstart + 3*lineW + _W + _U + (y*Bx + x)]
	
				#### activ(_sum)
				f0 = 1 / (1 + exp(-f0))
				f1 = 1 / (1 + exp(-f1))
				f2 = 1 / (1 + exp(-f2))
				g0 = tanh(g0)
	
				#### e = f0 * e[-1] + f1 * g0
				#### l - 1 >= 0
				e_1 = (var[total*(time-1) + ystart + (y*Bx + x)] if time > 0 else 0)
				e = f0*e_1 + f1*g0

				h = f2 * e
	
				var[total*time + ystart + (y*Bx + x)] = e
				var[total*time + ystart + Bx*Ay + (y*Bx + x)] = h
	
	'''
	=========== Forward ==================
	f0 = f(sf0 = xW + h[-1]U + b)
	f1 = f(sf1 = xW + h[-1]U + b)
	f2 = f(sf2 = xW + h[-1]U + b)
	g0 = f(sg0 = xW + h[-1]U + b)
	e = f0*e[-1] + f1*g0
	h = f2 * e
	
	========== Backward propagation (chain derivation) ============
	
	#h = f2 * e
		grad(f2) = grad(h) * e
		grad(e) += grad(h) * f2
	#e = f0*e[-1] + f1*g0
		grad(f0) = grad(e) * e[-1]
		grad(e[-1]) += grad(e) * f0
		grad(f1) = grad(e) * g0
		grad(g0) = grad(e) * f1
	#f0 = f(sf0 = xW + h[-1]U + b)
		grad(sf0) = grad(f0) * f'(sf0) = grad(f0) * f0 * (1 - f0)		#f = logistic; f' = logistic' = f*(1 - f)
	#f1 = f(sf1 = xW + h[-1]U + b)
		grad(sf1) = grad(f1) * f'(sf1) = grad(f1) * f1 * (1 - f1)
	#f2 = f(sf2 = xW + h[-1]U + b)
		grad(sf2) = grad(f2) * f'(sf2) = grad(f2) * f2 * (1 - f2)
	#g0 = f(sg0 = xW + h[-1]U + b)
		grad(sg0) = grad(g0) * f'(sg0) = grad(g0) * (1 - g0*g0)		#g = tanh; g' = tanh' = 1 - g^2
	
	=============== A shorter version ===============
	(the grad(e) have to be computed firste, because it's value will be use several times)
	
	grad(e) += grad(h) * f2
	grad(e[-1]) += grad(e) * f0
	grad(sf0) = grad(e) * e[-1] * f0 * (1 - f0) 
	grad(sf1) = grad(e) * g0 * f1 * (1 - f1)
	grad(sf2) = grad(h) * e * f2 * (1 - f2)
	grad(sg0) = grad(e) * f1 * (1 - g0*g0)
	
	================ C version ======================
	
	dh = grad[h]
	de = dh * f2 + grad[e]		#[time+1] had updated the gradient (we too for the line [time-1])
	
	grad[e] = de
	
	grad[e[-1] += de * f0
	dsf0 = de * e[-1] * f0 * (1 - f0)
	dsf1 = de * g0 * f1 * (1 - f1)
	dsf2 = dh * e * f2 * (1 - f2)
	dsg0 = de * f1 * (1 - g0*g0)
	'''
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, time:int, 
		w:[float], var:[float], locd:[float]):
		
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
	
		inp = sets*total*time + _set*total + istart
		W = _set*ws + wstart
		out = sets*total*time + _set*total + ystart
		locdpos = locds*sets*time + locds*_set + locdstart
	
		_W = Bx*Ax
		_U = Bx*Bx
		_B = Bx*Ay
	
		lineW = _W + _U + _B
	
		for y in range(Ay):
			for x in range(Bx):
				#### f0,f1,f2 = f(x@W + h[-1]@W + b)
				f0,f1,f2,g0 = 0, 0, 0, 0
				####
				# f0 is f0<x,y>
				####
	
				#### .W
				for k in range(Ax):
					vpos = inp + y*Ax + k
					wpos = (k*Bx + y)

					f0 += var[vpos]*w[W + 0*lineW + wpos]
					f1 += var[vpos]*w[W + 1*lineW + wpos]
					f2 += var[vpos]*w[W + 2*lineW + wpos]
					g0 += var[vpos]*w[W + 3*lineW + wpos]
	
				#### .U
				if time > 0:
					for k in range(Bx):
						#out == t
						#out - total*sets == sets*total*(l-1) + _set*total + istart
						vpos = total*sets*(time-1) + _set*total + ystart + (Bx*Ay) + (y*Bx + k) 
						wpos = _W + (k*Bx + y)

						f0 += var[vpos]*w[W + 0*lineW + wpos]
						f1 += var[vpos]*w[W + 1*lineW + wpos]
						f2 += var[vpos]*w[W + 2*lineW + wpos]
						g0 += var[vpos]*w[W + 3*lineW + wpos]
	
				#### .B
				wpos = _W + _U + y*Bx + x
				f0 += w[W + 0*lineW + wpos]
				f1 += w[W + 1*lineW + wpos]
				f2 += w[W + 2*lineW + wpos]
				g0 += w[W + 3*lineW + wpos]
	
				#### activate( _sum )
				f0 = 1 / (1 + exp(-f0))
				f1 = 1 / (1 + exp(-f1))
				f2 = 1 / (1 + exp(-f2))
				g0 = tanh(g0)
	
				#### e = f0 * e[-1] + f1 * g0
				#### l-1 >= 0
				e_1 = (var[sets*total*(time-1) + _set*total + ystart + y*Bx + x] if time > 0 else 0)
	
				e = f0*e_1 + f1*g0
				h = f2 * e
	
				locd_f0 = f0#f2*e_1*( f0*(1 - f0) )
				locd_f1 = f1#f2*g0*( f1*(1 - f1) )
				locd_f2 = f2#e*( f2*(1 - f2) )
				locd_g0 = g0#f2*f1*( 1 - g0*g0)
	
				var[out + (y*Bx + x)] = e
				var[out + Bx*Ay + (y*Bx + x)] = h
	
				locd[locdpos + 0*Bx*Ay + (y*Bx + x)] = locd_f0
				locd[locdpos + 1*Bx*Ay + (y*Bx + x)] = locd_f1
				locd[locdpos + 2*Bx*Ay + (y*Bx + x)] = locd_f2
				locd[locdpos + 3*Bx*Ay + (y*Bx + x)] = locd_g0
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, time:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
	
		inp = sets*total*time + _set*total + istart
		W = _set*ws + wstart
		out = sets*total*time + _set*total + ystart
		locdpos = locds*sets*time + locds*_set + locdstart
	
		_W = Bx*Ax
		_U = Bx*Bx
		_B = Bx*Ay
	
		lineW = _W + _U + _B
	
		Yx = Bx
		Yy = Ay
	
		for y in range(Ay):
			for x in range(Bx):
				epos = out + (y*Bx + x)
				e_1pos = total*sets*(time-1) + total*_set + ystart + (y*Bx + x) #if l == 0 , e_1pos <= 0
				hpos = out + Bx*Ay + (y*Bx + x)	#Bx*Ay is the space of `e`

				e = var[epos]
	
				dH = grad[hpos]
	
				f0 = locd[locdpos + 0*Bx*Ay + (y*Bx + x)] * dH
				f1 = locd[locdpos + 1*Bx*Ay + (y*Bx + x)] * dH
				f2 = locd[locdpos + 2*Bx*Ay + (y*Bx + x)] * dH
				g0 = locd[locdpos + 3*Bx*Ay + (y*Bx + x)] * dH
	
				de = grad[epos] + dH * f2		#grad(e) += dH*f2
	
				grad[epos] = de
	
				if time > 0:
					grad[e_1pos] += de * f0
				dsf0 = de * var[e_1pos] * f0 * (1 - f0)
				dsf1 = de * g0 * f1 * (1 - f1)
				dsf2 = dH * e * f2 * (1 - f2)
				dsg0 = de * f1 * (1 - g0*g0)
	
				#### .W
				for k in range(Ax):
					vpos = inp + y*Ax + k
					wpos = (k*Bx + x)
	
					#f0 += var[vpos]*w[W + wpos]
					grad[vpos] += dsf0 * w[W + 0*lineW + wpos]
					meand[W + 0*lineW + wpos] += dsf0 * var[vpos]
	
					#f1 += var[vpos]*w[W + lineW + wpos]
					grad[vpos] += dsf1 * w[W + 1*lineW + wpos]
					meand[W + 1*lineW + wpos] += dsf1 * var[vpos]
	
					#f2 += var[vpos]*w[W + 2*lineW + wpos]
					grad[vpos] += dsf2 * w[W + 2*lineW + wpos]
					meand[W + 2*lineW + wpos] += dsf2 * var[vpos]
	
					#g0 += var[vpos]*w[W + 3*lineW + wpos]
					grad[vpos] += dsg0 * w[W + 3*lineW + wpos]
					meand[W + 3*lineW + wpos] += dsg0 * var[vpos]
	
				#### .U
				if time > 0:
					for k in range(Bx):
						#out == t
						#out - total*sets == sets*total*(l-1) + _set*total + istart
						vpos = sets*total*(time-1) + _set*total + ystart + Bx*Ax + (y*Ax + k) 	#h[-1][y][x]
						wpos = _W + (k*Bx + y)
	
						#f0 += var[vpos]*w[W + wpos]
						grad[vpos] += dsf0 * w[W + 0*lineW + wpos]
						meand[W + 0*lineW + wpos] += dsf0 * var[vpos]
	
						#f1 += var[vpos]*w[W + lineW + wpos]
						grad[vpos] += dsf1 * w[W + 1*lineW + wpos]
						meand[W + 1*lineW + wpos] += dsf1 * var[vpos]
	
						#f2 += var[vpos]*w[W + 2*lineW + wpos]
						grad[vpos] += dsf2 * w[W + 2*lineW + wpos]
						meand[W + 2*lineW + wpos] += dsf2 * var[vpos]
	
						#g0 += var[vpos]*w[W + 3*lineW + wpos]
						grad[vpos] += dsg0 * w[W + 3*lineW + wpos]
						meand[W + 3*lineW + wpos] += dsg0 * var[vpos]
	
				#### .B
				wpos = _W + _U + (y*Bx + x)
	
				meand[W + 0*lineW + wpos] += dsf0
				meand[W + 1*lineW + wpos] += dsf1
				meand[W + 2*lineW + wpos] += dsf2
				meand[W + 3*lineW + wpos] += dsg0
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
		return 2*Bx*Ay
	
	def buildstackmodel_weights(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
		return 4*(Bx*Ax + Bx*Bx + Bx*Ay)
	
	def buildstackmodel_locds(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
		return 4*Bx*Ay
	
	#### Labels Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
		return [(f'{_id}.e [lstm2d]',stack_start), (f'{_id}.h [lstm2d]',stack_start+Ax*Ay)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
		wline = Bx*Ax + Bx*Bx + Bx*Ay
		return [
			(f'{_id}.Wf0 [lstm2d]',stack_start),(f'{_id}.Uf0 [lstm2d]',stack_start+Bx*Ax),(f'{_id}.Bf0 [lstm2d]',stack_start+Bx*Ax+Bx*Bx),
			(f'{_id}.Wf1 [lstm2d]',stack_start+wline),(f'{_id}.Uf1 [lstm2d]',stack_start+wline+Bx*Ax),(f'{_id}.Bf1 [lstm2d]',stack_start+wline+Bx*Ax+Bx*Bx),
			(f'{_id}.Wf2 [lstm2d]',stack_start+2*wline),(f'{_id}.Uf2 [lstm2d]',stack_start+2*wline++Bx*Ax),(f'{_id}.Bf2 [lstm2d]',stack_start+2*wline+Bx*Ax+Bx*Bx),
			(f'{_id}.Wg0 [lstm2d]',stack_start+3*wline),(f'{_id}.Ug0 [lstm2d]',stack_start+3*wline+Bx*Ax),(f'{_id}.Bg0 [lstm2d]',stack_start+3*wline+Bx*Ax+Bx*Bx)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = self.params
		return [(f'{_id}.f0 [lstm2d]',stack_start),(f'{_id}.f1 [lstm2d]',stack_start+Bx*Ay),(f'{_id}.f2 [lstm2d]',stack_start+2*Bx*Ay),(f'{_id}.g0 [lstm2d]',stack_start+3*Bx*Ay)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_lstm2d = "Ax", "Ay", "Bx", "drate"
	
	requiredposition_lstm2d = 1,1,1,0,0,0,0,1
	
	def setupparamsstackmodel(self, istart, ystart, wstart, lstart, required):
		Ax,Ay,Bx,drate = required
		return Ax,Ay,Bx,istart,ystart,wstart,lstart,drate

	### Check Params Input output

	def need_inputs(self, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		Ax,Ay,Bx,drate = required
		return Ax*Ay

	def check_input_output(self, last_vars, required):	#verifier que le vars du l'instruction precedente est de la meme taille 
		Ax,Ay,Bx,drate = required
		assert last_vars == self.need_inputs(required)