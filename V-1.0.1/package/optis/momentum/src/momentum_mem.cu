#include "package/optis/momentum/head/momentum.cuh"

float opti_momentum_alpha = 1e-5;
float opti_momentum_moment = 1 - 1e-5;

//	Minimum echopes to test the potential
uint MOMENTUM_min_echopes = 2;

void * MOMENTUM_space_mk(Opti_t * opti) {
	float * ret_d;

	SAFE_CUDA(cudaMalloc((void**)&ret_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMemset(ret_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	return (void*)ret_d;
};

void MOMENTUM_free(Opti_t * opti) {
	SAFE_CUDA(cudaFree((float*)opti->opti_space))
};

void MOMENTUM_set_one_arg(Opti_t * opti, char * name, char * value) {
	if (strcmp(name, "ALPHA") == 0) {
		opti_momentum_alpha = atof(value);
	} else if (strcmp(name, "MOMENT") == 0) {
		opti_momentum_moment = atof(value);
	} else {
		ERR("What is %s (of value %s)", name, value);
	}
};

const char * MOMENTUM_CONST_ARRAY[MOMENTUM_CONSTS] = {"ALPHA", "MOMENT"};
const uint MOMENTUM_CONSTS_AMOUNT = MOMENTUM_CONSTS;