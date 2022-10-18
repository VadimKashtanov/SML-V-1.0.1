from package.insts.sum.py.sum import SUM
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_SUM(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []

	mdl = Mdl(
		[
			SUM([size:=5, items:=3, 0, size*items])
	
		],
		inputs:=size*items,
		outputs:=size,
		_vars:=size,
		w:=[random() for _ in range(0)],
		locds:=0,

		vsep := [('sum.input',0), ('sum.output',inputs)],
		wsep := [],
		lsep := []
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
		[],	#adadelta
		[]	#adamax
	]

	gtics_args = [
		[('elites', '2')],	#elite
		[],					#genetique
	]