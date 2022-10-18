from kernel.py.mdl import Mdl
from kernel.py.data import Data
from kernel.py.use import Use
from kernel.py.train import Train
from kernel.py.optis import Opti_Class

from random import random, seed
import struct as st

class Test_MDL:
	score_algo = None
	score_class = None
	score_consts = None

	mdl = None

	lines = None
	sets = None

	def check(self):
		assert self.score_algo != None
		assert self.score_class != None
		assert self.score_consts != None

		assert self.mdl != None
		assert self.lines != None
		assert self.sets != None

	def bins(self):
		bins = b''

		####	Le Model et le contenair de Données		####
		mdl = self.mdl
		mdl.check()
		data = Data(
			1, self.lines, 
			[random() for _ in range(self.lines * mdl.inputs)], 
			[random() for _ in range(self.lines * mdl.outputs)])
		data.check()

		#
		bins +=	mdl.bins() + st.pack('I',123)
		bins += data.bins() + st.pack('I',123)

		#### Test du Cpu_t & Use_t ####

		use = Use(mdl, data)
		use.set_inputs()
		use.forward()

		#
		bins += use.bins() + st.pack('I',123)

		#### Test du Train_t (l'initialisation et le forward) ####

		train = Train(mdl, data, self.sets)
		train.randomize(0)
		train.check()

		bins += st.pack('I', self.sets) + st.pack('I',123)

		opti_class = Opti_Class(train, self.score_class, None)
		opti_class.score.check()

		#Set args
		for arg,value in self.score_consts:
			opti_class.score.CONSTS[arg] = int(value)

		bins += st.pack('I', self.score_algo) + st.pack('I',123)	#score_algo (meansquared=0, softmaxcrossentropy=1)
		bins += opti_class.score.bins() + st.pack('I',123)

		#######################

		train.set_inputs(0)
		train.null_grad_meand()
		train.forward()
		#
		bins += train.bin_w() + st.pack('I',123)
		bins += train.bin_v() + st.pack('I',123)
		bins += train.bin_l() + st.pack('I',123)
		#
		opti_class.dloss()
		train.backward()
		#
		bins += train.bin_g() + st.pack('I',123)
		bins += train.bin_m() + st.pack('I',123)

		#### On retourn le binaire qui sera plus tard ajouté au fichier 'save.bin' ####
		return bins

class Test_SCORE:
	score_algo = None
	score_class = None
	score_consts = None

	mdl = None

	lines = None
	sets = None

	def check(self):
		assert self.score_algo != None
		assert self.score_class != None
		assert self.score_consts != None

		assert self.mdl != None

		assert self.lines != None
		assert self.sets != None
		
	def bins(self):
		bins = b''

		####	Le Model et le contenair de Données		####
		mdl = self.mdl
		mdl.check()
		data = Data(
			1, self.lines, 
			[random() for _ in range(self.lines * mdl.inputs)], 
			[random() for _ in range(self.lines * mdl.outputs)])
		data.check()

		#
		bins +=	mdl.bins() + st.pack('I',123)
		bins += data.bins() + st.pack('I',123)

		#### Loss & Dloss ####

		train = Train(mdl, data, self.sets)
		train.randomize(0)
		train.check()

		bins += st.pack('II', self.sets, 123)

		opti_class = Opti_Class(train, self.score_class, None)
		opti_class.score.check()

		#Set args
		for arg,value in self.score_consts:
			opti_class.score.CONSTS[arg] = int(value)

		bins += st.pack('I', self.score_algo) + st.pack('I',123)	#score_algo (meansquared=0, softmaxcrossentropy=1)
		bins += opti_class.score.bins() + st.pack('I',123)

		#loss,dloss

		train.set_inputs(0)#batch=0
		train.null_grad_meand()
		train.forward()

		#dloss
		opti_class.dloss()
		train.backward()	#ici on va verifier toute l'array de train->_grad
		bins += train.bin_g() + st.pack('I',123)

		#loss
		opti_class.loss()	#ici on va verifier que la partie `output` de train->_grad est bien le bon loss(g,w), car la partie input et vars non output n'a pas bouge (car non efface)
		bins += train.bin_g() + st.pack('I',123)
		bins += opti_class.bin_score() + st.pack('I',123)	#puis on verifie que le score systeme est le bon

		#
		return bins

class Test_OPTI:
	score_algo = None
	score_class = None
	score_consts = None

	opti_algo = None
	opti_class = None
	opti_consts = None

	mdl = None

	lines = None
	sets = None

	def check(self):
		assert self.score_algo != None
		assert self.score_class != None
		assert self.score_consts != None

		assert self.opti_algo != None
		assert self.opti_class != None
		assert self.opti_consts != None

		assert self.mdl != None

		assert self.lines != None
		assert self.sets != None
		
	def bins(self):
		bins = b''

		####	Le Model et le contenair de Données		####
		mdl = self.mdl
		mdl.check()
		data = Data(
			1, self.lines, 
			[random() for _ in range(self.lines * mdl.inputs)], 
			[random() for _ in range(self.lines * mdl.outputs)])
		data.check()

		#
		bins +=	mdl.bins() + st.pack('I',123)
		bins += data.bins() + st.pack('I',123)

		#### 	Optimizer 	####

		train = Train(mdl, data, self.sets)
		train.randomize(0)
		train.check()

		bins += st.pack('II', self.sets, 123)

		opti_class = Opti_Class(train, self.score_class, self.opti_class)
		opti_class.check()

		#Set args
		for arg,value in self.score_consts:
			opti_class.score.CONSTS[arg] = float(value)

		for arg,value in self.opti_consts:
			opti_class.opti.CONSTS[arg] = float(value)

		bins += st.pack('III', self.score_algo, self.opti_algo, 123)
		bins += opti_class.score.bins() + st.pack('I',123)	#consts
		bins += opti_class.opti.bins() + st.pack('I',123)	#consts

		#loss,dloss
		bins += st.pack('II', opti_class.opti.MIN_TEST_ECHOPES, 123)
		for l in range(opti_class.opti.MIN_TEST_ECHOPES):
			train.set_inputs(0)	#batch=0
			train.null_grad_meand()
			train.forward()
			opti_class.dloss()
			train.backward()
			opti_class.optimize()

		bins += train.bin_m() + st.pack('I',123)	#pour verifier si au bout de la seconde boucle ou plus c'est bon
		bins += train.bin_w() + st.pack('I',123)

		#
		return bins

def test_package(TEST_MODELS, TEST_SCORES, TEST_OPTIS):
	'''On test tout le kernel et le package

	Les 4 TEST_X sont des listes avec les class définies au dessus.
	Elle testent chaqu'une un element comme une instruction, un optimizeur ...

	Le but est de tester tout le code C en consederant que le code python est correcte (ca peut etre aussi l'inverse).
		
	Techniquement si je test juste scores, opti et gtics, si il n'y a aucune erreur, alors test_models aussi ne doit avoire aucune erreure
	Mais pour debbug il faut test_models pour correctement voir les erreures et les corriger facilement.
	'''

	bins = b''

	for test_things in (TEST_MODELS, TEST_SCORES, TEST_OPTIS):

		bins += st.pack('I', len(test_things))
		for TEST in test_things:
			test = TEST()
			test.check()
			bins += test.bins()

	return bins