#include "package/optis/optis.cuh"

uint OPTI_MIN_ECHOPES[OPTIS] = {
	SGD_min_echopes,
	MOMENTUM_min_echopes,
	RMSPROP_min_echopes,
	ADAM_min_echopes,
};

void* (*OPTI_OPTI_SPACE_MK_ARRAY[OPTIS])(Opti_t * opti) = {
	SGD_space_mk,
	MOMENTUM_space_mk,
	RMSPROP_space_mk,
	ADAM_space_mk,
};

void (*OPTI_OPTI_SET_ONE_ARG_ARRAY[OPTIS])(Opti_t * opti, char * name, char * value) = {
	SGD_set_one_arg,
	MOMENTUM_set_one_arg,
	RMSPROP_set_one_arg,
	ADAM_set_one_arg,
};

void (*OPTI_OPTIMIZE_ARRAY[OPTIS])(Opti_t * opti) = {
	SGD_optimize,
	MOMENTUM_optimize,
	RMSPROP_optimize,
	ADAM_optimize,
};

void (*OPTI_FREE_OPTI_ARRAY[OPTIS])(Opti_t * opti) = {
	SGD_optimize,
	MOMENTUM_optimize,
	RMSPROP_optimize,
	ADAM_optimize,
};

const uint OPTI_CONST_AMOUNT[OPTIS] = {
	SGD_CONSTS_AMOUNT,
	MOMENTUM_CONSTS_AMOUNT,
	RMSPROP_CONSTS_AMOUNT,
	ADAM_CONSTS_AMOUNT,
};

const char ** OPTI_CONST_ARRAY[OPTIS] = {
	SGD_CONST_ARRAY,
	MOMENTUM_CONST_ARRAY,
	RMSPROP_CONST_ARRAY,
	ADAM_CONST_ARRAY,
};

