#pragma once

#include "kernel/head/optis.cuh"

void * CROSSENTROPY_space_mk(Opti_t * opti);
void CROSSENTROPY_free(Opti_t * opti);

void CROSSENTROPY_set_one_arg(Opti_t * opti, char * name, char * value);

void CROSSENTROPY_dloss(Opti_t * opti);
void CROSSENTROPY_loss(Opti_t * opti);

#define CROSSENTROPY_CONSTS 0
extern const char * CROSSENTROPY_CONST_ARRAY[CROSSENTROPY_CONSTS];
extern const uint CROSSENTROPY_CONSTS_AMOUNT;