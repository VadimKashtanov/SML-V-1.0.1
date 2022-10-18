#DOT2D = 1

from package.insts.kconvl2d.py.kconvl2d import KCONVL2D
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_KCONVL2D(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	'''mdl = Mdl(
		[
			KCONVL2D([Ax:=8, Ay:=6, Kx:=3, Ky:=5, n0:=2, n1:=3, strideX:=1, strideY:=2, paddingX:=2, paddingY:=1, activ:=3, istart:=0, ystart:=Ax*Ay*n0, wstart:=0, locdstart:=0, drop_rate:=0])
		],
		inputs:=Ax*Ay*n0,
		outputs:=int((Ax-2*paddingX)/strideX) * int((Ay-2*paddingY)/strideY) * n1,
		_vars:=outputs,
		w:=[random() for _ in range(Kx*Ky*n0*n1 + outputs)],
		locds:=outputs,

		vsep := [('kconvl2d.input',0), ('kconvl2d.output',inputs)],
		wsep := [('kconvl2d.K',0), ('kconvl2d.B',Kx*Ky*n0*n1)],
		lsep := [('kconvl2d.y_locd',0)]
	)'''

	mdl = Mdl(
		[
			KCONVL2D([Ax:=8, Ay:=6, Kx:=3, Ky:=5, n0:=2, n1:=3, strideX:=1, strideY:=2, paddingX:=2, paddingY:=1, activ:=3, istart:=0, ystart:=Ax*Ay*n0, wstart:=0, locdstart:=0, drop_rate:=0])
		],
		inputs:=Ax*Ay*n0,
		outputs:=int((Ax-2*paddingX)/strideX) * int((Ay-2*paddingY)/strideY) * n1,
		_vars:=outputs,
		w:=[random() for _ in range(Kx*Ky*n0*n1 + outputs)],
		locds:=outputs,

		vsep := [('kconvl2d.input',0), ('kconvl2d.output',inputs)],
		wsep := [('kconvl2d.K',0), ('kconvl2d.B',Kx*Ky*n0*n1)],
		lsep := [('kconvl2d.y_locd',0)]
	)

	lines = 2
	sets = 4

	scores_args = [
		[],	#mean squared
		[]	#cross entropy
	]

	optis_args = [
		[],	#sgd
		[],	#momentum
		[],	#rmsprop
		[],	#adam
	]

	gtics_args = [
		[('elites', '2')],	#elite
		[],					#genetique
	]