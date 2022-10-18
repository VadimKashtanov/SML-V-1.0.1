#DOT1D = 0

from package.insts.pool2dmax.py.pool2dmax import POOL2DMAX
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_POOL2DMAX(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	'''mdl = Mdl(
		[
			# ['Ax','Ay', 'Xpool', 'Ypool', 'input_start','ystart','locdstart']
			POOL2DMAX([Ax:=9,Ay:=4, Xpool:=3, Ypool:=2, 0, Ax*Ay, 0])
		],
		inputs:=Ax*Ay,
		outputs:=int(Ax/Xpool)*int(Ay/Ypool),
		_vars:=int(Ax/Xpool)*int(Ay/Ypool),
		w:=[],
		locds:=int(Ax/Xpool)*int(Ay/Ypool),

		vsep := [('pool2dmax.input',0), ('pool2dmax.output',Ax*Ay)],
		wsep := [],
		lsep :=	[('pool2dmax.y_locd',0)]
	)'''

	mdl = Mdl(
		[
			# ['Ax','Ay', 'Xpool', 'Ypool', 'input_start','ystart','locdstart']
			POOL2DMAX([Ax:=9,Ay:=4, Xpool:=3, Ypool:=2, 0, Ax*Ay, 0])
		],
		inputs:=Ax*Ay,
		outputs:=int(Ax/Xpool)*int(Ay/Ypool),
		_vars:=int(Ax/Xpool)*int(Ay/Ypool),
		w:=[],
		locds:=int(Ax/Xpool)*int(Ay/Ypool),

		vsep := [('pool2dmax.input',0), ('pool2dmax.output',Ax*Ay)],
		wsep := [],
		lsep :=	[('pool2dmax.y_locd',0)]
	)

	lines = 2
	sets = 4#8

	#2 elites => 8-2=6 free sets => 6/2 = 3 clones, so 2 elites and 3 clones each. (comme ça ça fait 2 chiffres differents)

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