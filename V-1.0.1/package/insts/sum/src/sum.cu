#include "package/insts/sum/head/sum.cuh"

void sum_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
};

void sum_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;
	uint total = mdl->total;
	uint * param = mdl->param[inst];

	uint size   = param[0],	\
		 items  = param[1],	\
		 istart = param[2],	\
		 ystart = param[3];

	float _sum;
	
	float * var = cpu->var;

	for (uint i=0; i < size; i++) {
		_sum = 0;
		for (uint j = 0; j < items; j++) {
			_sum += var[time*total + istart + j*size + i];
		}

		var[time*total + ystart + i] = _sum;
	}
};

void sum_use(Use_t * use, uint inst, uint time) {
	sum_use_call_mode_th11(use, inst, time);
};

void sum_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	sum_forward_call_mode_th11(train, inst, time, start_seed);
};

void sum_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	sum_backward_call_mode_th11(train, inst, time, start_seed);
};