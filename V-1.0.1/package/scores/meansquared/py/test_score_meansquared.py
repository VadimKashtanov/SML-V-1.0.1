from package.insts.dot1d.py.dot1d import DOT1D
from package.scores.meansquared.py.meansquared import MEANSQUARED
from kernel.py.mdl import Mdl
from kernel.py.test_package import Test_MDL, Test_SCORE, Test_OPTI
from random import random, seed

seed(0)

class TEST_SCORE_MEANSQUARED(Test_SCORE):
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

		vsep := [('input',0), ('output',Ax)],
		wsep := [('W',0),('B',Ax*Yx)],
		lsep :=	[('y locd',0)]
	)

	lines = 2
	sets = 3

#1 - Ligue De La Sauvgarde De L'Ethnie Et Du Patrimoine Français
#2 - Ligue Pour L'Independance Et De La Defense de La Patrie
#3 - Alliance Eternelle Des Guerriers Libres de Gergovie
#4 - Aiu-Rios-Cingeto (Eternel Libre Guerrier)
#5 - Ver-Cingeto-rix (grand guerrier roi), Brennos (chef)

#6 - Alliance Europeene De L'Ethnie Blanche
#7 - Agents De La Reimigration