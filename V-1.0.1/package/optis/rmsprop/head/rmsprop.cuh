#pragma once

#include "kernel/head/optis.cuh"

extern float opti_rmsprop_alpha;// = 1e-5;
extern float opti_rmsprop_beta;// = 1e-4;

//	Minimum echopes to test the potential
extern uint RMSPROP_min_echopes;// = 2;

/*
Root Mean Squared Propagation

	v = beta * v + (1-beta) * grad(w)^2
	w -= alpha * grad(w) / sqrt(v)
*/
	
void * RMSPROP_space_mk(Opti_t * opti);
void RMSPROP_free(Opti_t * opti);

void RMSPROP_set_one_arg(Opti_t * opti, char * name, char * value);

__global__
void RMSPROP_kernel_th11(
	float alpha, float beta,
	uint weights,
	float * v, float * weight, float * meand);

void RMSPROP_optimize(Opti_t * opti);

#define RMSPROP_CONSTS 2
extern const char * RMSPROP_CONST_ARRAY[RMSPROP_CONSTS];// = {"ALPHA", "BETA"};
extern const uint RMSPROP_CONSTS_AMOUNT;