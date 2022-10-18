from package.insts.softmax.py.softmax import SOFTMAX
from package.scores.crossentropy.py.crossentropy import CROSSENTROPY
from kernel.py.mdl import Mdl
from kernel.py.test_package import Test_MDL, Test_SCORE, Test_OPTI
from random import random, seed

seed(0)

class TEST_SCORE_CROSSENTROPY(Test_SCORE):
	score_algo = 1
	score_class = CROSSENTROPY
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
	sets = 3

#1 - Ligue De La Sauvgarde De L'Ethnie Et Du Patrimoine Français
#2 - Ligue Pour L'Independance Et De La Defense de La Patrie
#3 - Alliance Eternelle Des Guerriers Ligres De Gaule
#4 - Aiu-Rios-Cingeto (Eternel Libre Guerrier)
#5 - Ver-Cingeto-rix (grand guerrier roi), Brennos (chef)

#6 - Alliance Europeene De L'Ethnie Blanche
#7 - Agents De La Reimigration