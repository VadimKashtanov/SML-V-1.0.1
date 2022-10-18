#pragma once

#include "kernel/head/optis.cuh"

extern float opti_sgd_alpha;// = 1e-5;

//	Minimum echopes to test the potential
extern uint SGD_min_echopes;// = 1;

/*
Stocastic Gradient Descent.

Vannila or classic grandient descent. Only with a gradient step.

	w -= alpha * grad(w)
*/

typedef struct {
	uint echopes;
} SGDData_t;

void * SGD_space_mk(Opti_t * opti);
void SGD_free(Opti_t * opti);

void SGD_set_one_arg(Opti_t * opti, char * name, char * value);

__global__
void sgd_kernel_th11(
	float alpha,
	uint weights,
	float * weight, float * meand);

void SGD_optimize(Opti_t * opti);

#define SGD_CONSTS 1
extern const char * SGD_CONST_ARRAY[SGD_CONSTS];// = {"ALPHA"};
extern const uint SGD_CONSTS_AMOUNT;