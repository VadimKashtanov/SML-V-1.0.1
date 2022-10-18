#DOT2D = 1

from package.insts.gaussfiltre2d.py.gaussfiltre2d import GAUSSFILTRE2D
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_GAUSSFILTRE2D(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	'''mdl = Mdl(
		[
			GAUSSFILTRE2D([X:=5, Y:=7, 0,X*Y,0,0])
	
		],
		inputs:=X*Y,
		outputs:=X*Y,
		_vars:=X*Y,
		w:=[random() for _ in range(X)],
		locds:=X*Y,

		vsep := [('gaussfiltre2d.input',0), ('gaussfiltre2d.output',_len)],
		wsep := [('gaussfiltre2d.W',0)],
		lsep := [('gaussfiltre2d.y_locd',0)]
	)'''

	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=GAUSSFILTRE2D,
		required:=[X:=5, Y:=3]
	)

	lines = 2
	sets = 4#8

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