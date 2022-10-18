#include "package/insts/inst/head/inst.cuh"

void inst_check(uint * param) {
	//>0 <==> >= 1
	if (param[0] == 0) raise(SIGINT);
	if (param[1] >= 2) raise(SIGINT);
};

void inst_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;
	uint total = mdl->total;

	uint param0=mdl->param[inst][0],\
		 param1=mdl->param[inst][1];

	float * var = cpu->var;
	float * weight = mdl->weight;

	uint _inp=time*total + input_start,	\
		 _w=wstart;
	
	for (uint y=0; y < param0; y++) {
		
		var[time*mdl->total + ystart + y] = 0;
	}
};

void inst_use(Use_t * use, uint inst, uint time) {
	inst_use_call_mode_mod0(use, inst, time);
};

void inst_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	inst_forward_call_mode_mod0(train, inst, time, start_seed);
};

void inst_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	inst_backward_call_mode_mod0(train, inst, time, start_seed);
};