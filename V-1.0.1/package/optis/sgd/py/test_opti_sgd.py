from package.insts.dot1d.py.dot1d import DOT1D
from package.scores.meansquared.py.meansquared import MEANSQUARED
from package.optis.sgd.py.sgd import SGD
from kernel.py.mdl import Mdl
from kernel.py.test_package import Test_MDL, Test_SCORE, Test_OPTI
from random import random, seed

seed(0)

#		Laissez la Luxure aux faibles et aux parasites
#	Ver- Cingeto -rix		#Un Chef Des Guerriers Libres De Gaule et de Celtie
#	Rio- Cinget-os			#Guerrier Libre

class TEST_OPTI_SGD(Test_OPTI):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []

	opti_algo = 1
	opti_class = SGD
	opti_consts = [('ALPHA', 0.1)]

	mdl = Mdl(
		[
			DOT1D([Ax:=4,Yx:=3, 0, 0,Ax,0,0, 0])
		],
		inputs:=Ax,
		outputs:=Yx,
		_vars:=Yx,
		w:=[random() for _ in range(Ax*Yx + Yx)],
		locds:=Yx,

		vsep := [('input',0), ('output',Ax)],
		wsep := [('W',0),('B',Ax*Yx)],
		lsep :=	[('y locd',0)]
	)

	lines = 2
	sets = 3