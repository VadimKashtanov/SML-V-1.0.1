#include "package/scores/meansquared/head/meansquared.cuh"

void * MEANSQUARED_space_mk(Opti_t * opti) {
	return NULL;
};

void MEANSQUARED_free(Opti_t * opti) {

};

void MEANSQUARED_set_one_arg(Opti_t * opti, char * name, char * value) {
	ERR("There is no args for MEANSQUARED. (name=%s, value=%s)", name, value);
};

const char * MEANSQUARED_CONST_ARRAY[MEANSQUARED_CONSTS] = {};
const uint MEANSQUARED_CONSTS_AMOUNT = MEANSQUARED_CONSTS;