#pragma once

#include "kernel/head/optis.cuh"

extern float opti_adam_alpha;// = 1e-5;
extern float opti_adam_beta0;// = 1e-5;
extern float opti_adam_beta1;// = 1e-5;

//	Minimum echopes to test the potential
extern uint ADAM_min_echopes;// = 2;

/*Adaptive Moment Estimation (Adam)
	
	m = beta0*m + (1 - beta0)*grad(w)
	v = beta1*m + (1 - beta1)*grad(w)^2

	_m = m / ( 1 - beta0^t )
	_v = v / ( 1 - beta1^t )		t is echope
	
	w -= alpha * _m / sqrt(_v + eta)
*/

typedef struct {
	float * m_d, * v_d;		//_d == _device == vram == gpu ram = cudaMalloc

	uint echope;
} AdamData_t;

void * ADAM_space_mk(Opti_t * opti);
void ADAM_free(Opti_t * opti);

void ADAM_set_one_arg(Opti_t * opti, char * name, char * value);

__global__
void adam_kernel_th11(
	float beta0, float beta1, float alpha,
	uint echope,
	uint weights,
	float * m, float *v, float * weight, float * meand);

void ADAM_optimize(Opti_t * opti);

#define ADAM_CONSTS 3
extern const char * ADAM_CONST_ARRAY[ADAM_CONSTS];// = {"ALPHA", "BETA0", "BETA1"};
extern const uint ADAM_CONSTS_AMOUNT;