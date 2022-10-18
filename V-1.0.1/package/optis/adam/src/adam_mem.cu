#include "package/optis/adam/head/adam.cuh"

float opti_adam_alpha = 1e-5;
float opti_adam_beta0 = 1e-5;
float opti_adam_beta1 = 1e-5;

//	Minimum echopes to test the potential
uint ADAM_min_echopes = 2;

void * ADAM_space_mk(Opti_t * opti) {
	float * m_d, * v_d;

	SAFE_CUDA(cudaMalloc((void**)&m_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights));
	SAFE_CUDA(cudaMalloc((void**)&v_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights));

	SAFE_CUDA(cudaMemset(m_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights));
	SAFE_CUDA(cudaMemset(v_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights));

	AdamData_t * ret = (AdamData_t*)malloc(sizeof(AdamData_t));

	ret->m_d = m_d;
	ret->v_d = v_d;

	ret->echope = 0;

	return (void*)ret;
};

void ADAM_free(Opti_t * opti) {
	SAFE_CUDA(cudaFree(((AdamData_t*)opti->opti_space)->m_d))
	SAFE_CUDA(cudaFree(((AdamData_t*)opti->opti_space)->v_d))
	free(((AdamData_t*)opti->opti_space));
};

void ADAM_set_one_arg(Opti_t * opti, char * name, char * value) {
	if (strcmp(name, "ALPHA") == 0) {
		opti_adam_alpha = atof(value);
	} else if (strcmp(name, "BETA0") == 0) {
		opti_adam_beta0 = atof(value);
	} else if (strcmp(name, "BETA1") == 0) {
		opti_adam_beta1 = atof(value);
	} else {
		ERR("What is %s (of value %s)", name, value);
	}
};

const char * ADAM_CONST_ARRAY[ADAM_CONSTS] = {"ALPHA", "BETA0", "BETA1"};
const uint ADAM_CONSTS_AMOUNT = ADAM_CONSTS;