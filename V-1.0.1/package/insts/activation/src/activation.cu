#include "package/insts/activation/head/activation.cuh"

void activation_check(uint * param) {
	//>0 <==> >= 1
	if (param[0] == 0) raise(SIGINT);
	if (param[1] >= 4) raise(SIGINT);
};

void activation_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;
	uint total = mdl->total;

	uint _len=mdl->param[inst][0],			\
		 activ=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3];

	float * var = cpu->var;
	
	for (uint i=0; i < _len; i++) {
		float value = var[time*total + istart + i];

		if (activ == 0) value = 1 / (1 + exp(-value));
		else if (activ == 1) activ = tanh(value);
		else if (activ == 2) activ = exp(-value*value);
		else if (activ == 3) activ *= (activ > 0);

		var[time*total + ystart + i] = value;
	}
};

void activation_use(Use_t * use, uint inst, uint time) {
	activation_use_call_mode_th11(use, inst, time);
};

void activation_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	activation_forward_call_mode_th11(train, inst, time, start_seed);
};

void activation_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	activation_backward_call_mode_th11(train, inst, time, start_seed);
};