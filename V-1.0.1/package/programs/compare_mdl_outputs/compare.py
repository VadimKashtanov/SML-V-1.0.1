from package.package import INSTS_DICT, TEST_MODELS, TEST_SCORES, TEST_OPTIS
from kernel.py.test_package import test_package
from kernel.py.mdl import Mdl
from kernel.py.data import Data
from kernel.py.use import Use
from kernel.py.train import Train
from kernel.py.optis import Opti_Class

from random import random

from package.insts.fast_model import Fast_1Layer_FeedForward_Mdl, Fast_Feed_Forward_Mdl

mdl0 = Fast_1Layer_FeedForward_Mdl(
	inst:=INSTS_DICT['ACTIVATION'],
	required:=[_len:=4, activ:=1]
)

mdl1 = Fast_1Layer_FeedForward_Mdl(
	inst:=INSTS_DICT['ACTIVATION'],
	required:=[_len:=4, activ:=1]
)

#
#	Adding Mul instruction to make the lstm multiplication module
#

compare_floats = lambda a, b, p: (abs(a-b) < p)

if __name__ == "__main__":
	#	Check if inputs and outputs are same size for both
	assert mdl0.inputs == mdl1.inputs
	assert mdl0.outputs == mdl1.outputs

	inputs = mdl0.inputs
	outputs = mdl0.outputs

	lines = 3

	data = Data(
		batchs:=1, lines, 
		[random() for _ in range(lines * inputs)], 
		[random() for _ in range(lines * outputs)])

	out_array = [[], []]

	for i,mdl in enumerate([mdl0, mdl1]):
		mdl.check()
		data.check()

		use = Use(mdl, data)
		use.set_inputs()
		use.forward()

		outstart = mdl.total - mdl.outputs
		total = mdl.total

		out_array[i] = [use._var[line*total + outstart + i] for line in range(lines) for i in range(outputs)]

		del use

	for l in range(lines):
		print(f"\033[{(91 + l % 2)}m||\033[0m Line = {l} ###################")
		for o in range(outputs):
			v0 = out_array[0][l*outputs + o]
			v1 = out_array[1][l*outputs + o]
			
			if compare_floats(v0, v1, 0.0001):
				print(f"\033[{(l % 2 + 91)}m||\033[0m|| {o} |  \033[42m {v0} --- {v1} \033[0m")
			else:
				print(f"\033[{(l % 2 + 91)}m||\033[0m|| {o} |  \033[41m {v0} --- {v1} \033[0m")

	print("                 mdl0                   mdl1")