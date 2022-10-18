#pragma once

#include "kernel/head/optis.cuh"

extern float opti_momentum_alpha;// = 1e-5;
extern float opti_momentum_moment;// = 1 - 1e-5;

//	Minimum echopes to test the potential
extern uint MOMENTUM_min_echopes;// = 2;

/*
Stocastic Gradient Descent with momentum

Vannila or classic grandient descent. Only with a gradient step and momentum

	v = moment * v - alpha * grad(w)
	w += v
*/

void * MOMENTUM_space_mk(Opti_t * opti);
void MOMENTUM_free(Opti_t * opti);

void MOMENTUM_set_one_arg(Opti_t * opti, char * name, char * value);

__global__
void momentum_kernel_th11(
	float alpha, float moment,
	uint weights,
	float * v, float * weight, float * meand);

void MOMENTUM_optimize(Opti_t * opti);

#define MOMENTUM_CONSTS 2
extern const char * MOMENTUM_CONST_ARRAY[MOMENTUM_CONSTS];// = {"ALPHA", "MOMENT"};
extern const uint MOMENTUM_CONSTS_AMOUNT;