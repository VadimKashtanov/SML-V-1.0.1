#pragma once

#include "kernel/head/optis.cuh"

void * MEANSQUARED_space_mk(Opti_t * opti);
void MEANSQUARED_free(Opti_t * opti);

void MEANSQUARED_set_one_arg(Opti_t * opti, char * name, char * value);

void MEANSQUARED_dloss(Opti_t * opti);
void MEANSQUARED_loss(Opti_t * opti);

#define MEANSQUARED_CONSTS 0
extern const char * MEANSQUARED_CONST_ARRAY[MEANSQUARED_CONSTS];
extern const uint MEANSQUARED_CONSTS_AMOUNT;