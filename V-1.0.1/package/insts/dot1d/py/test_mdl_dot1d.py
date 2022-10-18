#DOT1D = 0

from package.insts.dot1d.py.dot1d import DOT1D
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_DOT1D(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	mdl = Mdl(
		[
			DOT1D([Ax:=4,Yx:=3, 0, 0,Ax,0,0, 0])
		],
		inputs:=Ax,
		outputs:=Yx,
		_vars:=Yx,
		w:=[random() for _ in range(Ax*Yx + Yx)],
		locds:=Yx,

		vsep := [('dot1d.input',0), ('dot1d.output',Ax)],
		wsep := [('dot1d.W',0),('dot1d.B',Ax*Yx)],
		lsep :=	[('dot1d.y_locd',0)]
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