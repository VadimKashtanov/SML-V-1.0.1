#pragma once

#include "kernel/head/train.cuh"

#include "pool2dmax_th11.cuh"

//['Ax','Ay', 'Xpool', 'Ypool', 'input_start','ystart','locdstart']

/*
	Add kernel to const (const_th11)

	And maybe texture (with inputs)
*/

void pool2dmax_check(uint * param);

//======================= Cpu_t forward ===========================

void pool2dmax_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void pool2dmax_use_call_mode_th11(Use_t * use, uint inst, uint time);

void pool2dmax_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void pool2dmax_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void pool2dmax_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void pool2dmax_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void pool2dmax_backward(Train_t * train, uint inst, uint time, uint start_seed);