#DOT1D = 0

from package.insts.pool2daverage.py.pool2daverage import POOL2DAVERAGE
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_POOL2DAVERAGE(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []

	mdl = Mdl(
		[
			# ['Ax','Ay', 'Xpool', 'Ypool', 'input_start','ystart']
			POOL2DAVERAGE([Ax:=9,Ay:=4, Xpool:=3, Ypool:=2, 0, Ax*Ay])
		],
		inputs:=Ax*Ay,
		outputs:=int(Ax/Xpool)*int(Ay/Ypool),
		_vars:=int(Ax/Xpool)*int(Ay/Ypool),
		w:=[],
		locds:=0,

		vsep := [('pool2daverage.input',0), ('pool2daverage.output', Ax*Ay)],
		wsep := [],
		lsep :=	[]
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