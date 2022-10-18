#SOFTMAX = 3

from package.insts.softmax.py.softmax import SOFTMAX
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_SOFTMAX(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []

	mdl = Mdl(
		[
			SOFTMAX([X:=3, 0, X])
	
		],
		inputs:=X,
		outputs:=X,
		_vars:= X,
		w:=[random() for _ in range(0)],
		locds:=0,

		vsep := [('softmax.input',0), ('softmax.output',X)],
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