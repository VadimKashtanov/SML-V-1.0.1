from kernel.py.mdl import Mdl
from random import random

class Fast_1Layer_FeedForward_Mdl(Mdl):
	def __init__(self, inst, required):
		inst = inst()

		inst.build_from_required(
			required, 
			inputs:=inst.need_inputs(required),
			istart:=0,
			ystart:=inputs,
			wstart:=0,
			lstart:=0
		)
		
		#	Computing params
		_vars = inst.buildstackmodel_vars()
		weights = inst.buildstackmodel_weights()
		locds = inst.buildstackmodel_locds()

		#	Labels
		_id = 0
		vsep = inst.labelstackmodel_vars(_id,ystart)
		wsep = inst.labelstackmodel_weights(_id,0)
		lsep = inst.labelstackmodel_locds(_id,0)

		super().__init__([inst], inputs, _vars, _vars, [random() for i in range(weights)], locds, vsep, wsep, lsep)

class Fast_Feed_Forward_Mdl(Mdl):
	def __init__(self, insts_required):
		'''
insts_required = [
	(Inst0, [req0, req1]),
	(Inst1, [req0, req1, req2, req3])
	...
]	'''
		insts = []
		inputs = insts_required[0][0]().need_inputs(insts_required[0][1])
		vsep, wsep, lsep = [], [], []

		istart = 0
		vars_stack = inputs
		weights_stack = 0
		locds_stack = 0

		for _id,(inst, required) in enumerate(insts_required):
			insts += [inst()]

			#inputs += [inst.need_inputs(required)]

			#vars_stack += inputs[-1]

			inst.build_from_required(
				required=required,
				inputs=inst.need_inputs(required),
				istart=istart,
				ystart=stack_start,
				wstart=weights_stack,
				lstart=locds_stack
			)

			vsep += inst.labelstackmodel_vars(   _id, vars_stack)
			wsep += inst.labelstackmodel_weights(_id, weights_stack)
			lsep += inst.labelstackmodel_locds(  _id, locds_stack)

			vars_stack += inst.buildstackmodel_vars()
			weights_stack += inst.buildstackmodel_weights()
			locds_stack += inst.buildstackmodel_locds()

			istart = vsep - inst.labelstackmodel_vars(   _id, vars_stack)

		super().__init__(insts=inst,
			inputs=inputs,
			outputs=insts[-1].buildstackmodel_vars(),
			_vars=vars_stack,
			w=[random() for i in range(weights_stack)],
			locds=locds_stack,
			vsep=vsep, wsep=wsep, lsep=lsep)