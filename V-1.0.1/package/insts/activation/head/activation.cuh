#pragma once

#include "kernel/head/train.cuh"

#include "activation_th11.cuh"

/*

activations = {
	1 / (1 + exp(-x))
	tanh(x)
	exp(-x*x)
	x * (x>=0)
}

F(x, activ) = activations[activ](x)

*/

//=========================== Sizes ===============================

/*
	inputs = Ax
	vars = Yx
	weights = Ax*Yx + Yx
	locds = Yx
*/

void activation_check(uint * param);

//======================= Cpu_t forward ===========================

void activation_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void activation_use_call_mode_th11(Use_t * use, uint inst, uint time);

void activation_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void activation_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void activation_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void activation_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void activation_backward(Train_t * train, uint inst, uint time, uint start_seed);