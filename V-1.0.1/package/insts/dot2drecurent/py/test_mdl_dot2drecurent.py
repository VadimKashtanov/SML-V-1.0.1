#DOT2D = 1

from package.insts.dot2drecurent.py.dot2drecurent import DOT2DRECURENT
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_DOT2DRECURENT(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	mdl = Mdl(
		[
			DOT2DRECURENT([Ax:=4,Ay:=3,At:=1,Bx:=2, 0, 0,Ax*Ay,0,0, 0])
	
		],
		inputs:=Ax*Ay,
		outputs:=Bx*Ay,
		_vars:=Bx*Ay,
		w:=[random() for _ in range(Ax*Bx + Bx*Ay)],
		locds:=Bx*Ay,

		vsep := [('dot2drecurent.input',0), ('dot2drecurent.output',Ax*Ay)],
		wsep := [('dot2drecurent.W',0), ('dot2drecurent.B',Ax*Bx)],
		lsep := [('dot2drecurent.y_locd',0)]
	)

	lines = 3
	sets = 2#8

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