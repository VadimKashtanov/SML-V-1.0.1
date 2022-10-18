#include "package/scores/scores.cuh"

void* (*OPTI_SCORE_SPACE_MK_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_space_mk,
	CROSSENTROPY_space_mk,
};

void (*OPTI_SCORE_SET_ONE_ARG_ARRAY[SCORES])(Opti_t * opti, char * name, char * value) = {
	MEANSQUARED_set_one_arg,
	CROSSENTROPY_set_one_arg,
};

void (*OPTI_COMPUTE_LOSS_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_loss,
	CROSSENTROPY_loss,
};

void (*OPTI_SCORES_DLOSS_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_dloss,
	CROSSENTROPY_dloss,
};

void (*OPTI_FREE_SCORE_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_free,
	CROSSENTROPY_free,
};

const uint SCORE_CONST_AMOUNT[SCORES] = {
	MEANSQUARED_CONSTS_AMOUNT,
	CROSSENTROPY_CONSTS_AMOUNT,
};

const char ** SCORE_CONST_ARRAY[SCORES] = {
	MEANSQUARED_CONST_ARRAY,
	CROSSENTROPY_CONST_ARRAY,
};

