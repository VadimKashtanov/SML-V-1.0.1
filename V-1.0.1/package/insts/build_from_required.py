from kernel.py.inst import Inst

class BuildFromRequired(Inst):
	def build_from_required(self, required, inputs, istart, ystart, wstart, lstart):
		self.check_input_output(inputs, required)
		self.params = self.setupparamsstackmodel(istart, ystart, wstart, lstart, required)
		return self