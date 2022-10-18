import struct as st

class Data:
	def __init__(self, batchs, lines, _input, output):
		self.batchs = batchs
		self.lines = lines
		self.inputs = int(len(_input)/(lines*batchs))
		self.outputs = int(len(output)/(lines*batchs))

		self.input = _input
		self.output = output

	def check(self):
		assert self.inputs * self.lines * self.batchs == len(self.input)
		assert self.outputs * self.lines * self.batchs == len(self.output)

	def bins(self):
		bins = st.pack('IIII', self.batchs, self.lines, self.inputs, self.outputs)

		bins += st.pack('f'*(self.batchs * self.lines * self.inputs), *self.input)
		bins += st.pack('f'*(self.batchs * self.lines * self.outputs), *self.output)

		return bins