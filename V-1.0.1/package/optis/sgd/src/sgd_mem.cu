#include "package/optis/sgd/head/sgd.cuh"

float opti_sgd_alpha = 0.1;//1e-3;

//	Minimum echopes to test the potential
uint SGD_min_echopes = 1;

void* SGD_space_mk(Opti_t * opti) {
	SGDData_t * ret = (SGDData_t*)malloc(sizeof(SGDData_t));
	ret->echopes = 0;
	return (void*)ret;
};

void SGD_free(Opti_t * opti) {
	free((SGDData_t*)opti->opti_space);
};

void SGD_set_one_arg(Opti_t * opti, char * name, char * value) {
	if (strcmp(name, "ALPHA") == 0) {
		opti_sgd_alpha = atof(value);
	} else {
		ERR("What is %s (of value %s)", name, value);
	}
};

const char *SGD_CONST_ARRAY[SGD_CONSTS] = {"ALPHA"};
const uint SGD_CONSTS_AMOUNT = SGD_CONSTS;