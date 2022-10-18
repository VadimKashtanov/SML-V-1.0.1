#DOT2D = 1

from package.insts.lstm2d.py.lstm2d import LSTM2D
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.mdl import Mdl
from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl
from kernel.py.test_package import Test_MDL
from random import random, seed

seed(0)

class TEST_MDL_LSTM2D(Test_MDL):
	score_algo = 0
	score_class = MEANSQUARED
	score_consts = []
	
	'''mdl = Mdl(
		[
			LSTM2D([Ax:=10, Ay:=5, Bx:=8, 0, Ax*Ay, 0, 0, 0])
		],
		inputs:=Ax*Ay,
		outputs:=2*Ax*Bx,
		_vars:=2*Ax*Bx,
		w:=[random() for _ in range(4*(Bx*Ax + Bx*Bx + Bx*Ay))],
		locds:=4*Bx*Ay,

		vsep := [('lstm2d.input',0), ('lstm2d.output',Ax*Ay)],
		wsep := [
			(f'Wf0 [lstm2d]',stack_start:=0),(f'Uf0 [lstm2d]',stack_start+Bx*Ax),(f'Bf0 [lstm2d]',stack_start+Bx*Ax+Bx*Bx),
			(f'Wf1 [lstm2d]',stack_start+(wline := Bx*Ax + Bx*Bx + Bx*Ay)),(f'Uf1 [lstm2d]',stack_start+wline+Bx*Ax),(f'Bf1 [lstm2d]',stack_start+wline+Bx*Ax+Bx*Bx),
			(f'Wf2 [lstm2d]',stack_start+2*wline),(f'Uf2 [lstm2d]',stack_start+2*wline+Bx*Ax),(f'Bf2 [lstm2d]',stack_start+2*wline+Bx*Ax+Bx*Bx),
			(f'Wg0 [lstm2d]',stack_start+3*wline),(f'Ug0 [lstm2d]',stack_start+3*wline+Bx*Ax),(f'Bg0 [lstm2d]',stack_start+3*wline+Bx*Ax+Bx*Bx)],
		lsep := [('lstm2d.y_locd',0)]
	)'''

	mdl = Fast_1Layer_FeedForward_Mdl(
		inst:=LSTM2D,
		required:=[Ax:=5, Ay:=3, Bx:=4, drate:=0]
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