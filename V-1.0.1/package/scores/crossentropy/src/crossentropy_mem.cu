#include "package/scores/crossentropy/head/crossentropy.cuh"

void * CROSSENTROPY_space_mk(Opti_t * opti) {
	return NULL;
};

void CROSSENTROPY_free(Opti_t * opti) {

};

void CROSSENTROPY_set_one_arg(Opti_t * opti, char * name, char * value) {
	ERR("There is no args for CROSSENTROPY. (name=%s, value=%s)", name, value);
};

const char * CROSSENTROPY_CONST_ARRAY[CROSSENTROPY_CONSTS] = {};
const uint CROSSENTROPY_CONSTS_AMOUNT = CROSSENTROPY_CONSTS;