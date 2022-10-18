#include "package/optis/rmsprop/head/rmsprop.cuh"

float opti_rmsprop_alpha = 1e-5;
float opti_rmsprop_beta = 1e-4;

//	Minimum echopes to test the potential
uint RMSPROP_min_echopes = 2;

void * RMSPROP_space_mk(Opti_t * opti) {
	float * v0_d;//, * v1_d;

	SAFE_CUDA(cudaMalloc((void**)&v0_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	//SAFE_CUDA(cudaMalloc((void**)&v1_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	SAFE_CUDA(cudaMemset(v0_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	//SAFE_CUDA(cudaMemset(v1_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	//RMSprop_data_t * ret = (RMSprop_data_t*)malloc(sizeof(RMSprop_data_t));

	//ret->v0_d = v0_d;
	//ret->v1_d = v1_d;

	//return (void*)ret;
	return (void*)v0_d;
};

void RMSPROP_free(Opti_t * opti) {
	//SAFE_CUDA(cudaFree((RMSprop_data_t*)opti->opti_space->v0_d))
	//SAFE_CUDA(cudaFree((RMSprop_data_t*)opti->opti_space->v1_d))
	//free((RMSprop_data_t*)opti->opti_space);
	SAFE_CUDA(cudaFree((float*)opti->opti_space))
};

void RMSPROP_set_one_arg(Opti_t * opti, char * name, char * value) {
	if (strcmp(name, "ALPHA") == 0) {
		opti_rmsprop_alpha = atof(value);
	} else if (strcmp(name, "BETA") == 0) {
		opti_rmsprop_beta = atof(value);
	} else {
		ERR("What is %s (of value %s)", name, value);
	}
};

const char * RMSPROP_CONST_ARRAY[RMSPROP_CONSTS] = {"ALPHA", "BETA"};
const uint RMSPROP_CONSTS_AMOUNT = RMSPROP_CONSTS;