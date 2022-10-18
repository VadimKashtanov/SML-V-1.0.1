from package.insts.activation.py.activation import ACTIVATION
from kernel.py.mdl import Mdl
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.test_package import Test_MDL
from random import random, seed
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl

seed(0)

class TEST_MDL_ACTIVATION(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	'''mdl = Mdl(
		[
			ACTIVATION([_len:=4, activ:=1, 0, 0,_len,0,0])
		],
		inputs:=_len,
		outputs:=_len,
		_vars:=_len,
		w:=[random() for _ in range(0)],
		locds:=_len,

		vsep := [('activation.input',0), ('activation.output',_len)],
		wsep := [],
		lsep :=	[('activation.y_locd',0)]
	)'''
	
	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=ACTIVATION,
		required:=[_len:=4, activ:=1]
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