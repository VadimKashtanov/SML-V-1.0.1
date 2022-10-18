#pragma once

#include "kernel/head/train.cuh"

#include "kconvl2d_th11.cuh"

//['Ax','Ay', 'Kx', 'Ky', 'n0', 'n1', 'strideX', 'strideY', 'paddingX', 'paddingY', 'activ', 'input_start','ystart','wstart','locdstart', 'drop_rate']

/*
	Add kernel to const (const_th11)

	And maybe texture (with inputs)
*/

void kconvl2d_check(uint * param);

//======================= Cpu_t forward ===========================

void kconvl2d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void kconvl2d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void kconvl2d_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void kconvl2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void kconvl2d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void kconvl2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void kconvl2d_backward(Train_t * train, uint inst, uint time, uint start_seed);