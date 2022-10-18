#pragma once

#include "kernel/head/train.cuh"

#include "pool2daverage_th11.cuh"

//['Ax','Ay', 'Xpool', 'Ypool', 'input_start','ystart']

/*
	Add kernel to const (const_th11)

	And maybe texture (with inputs)
*/

void pool2daverage_check(uint * param);

//======================= Cpu_t forward ===========================

void pool2daverage_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void pool2daverage_use_call_mode_th11(Use_t * use, uint inst, uint time);

void pool2daverage_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void pool2daverage_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void pool2daverage_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void pool2daverage_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void pool2daverage_backward(Train_t * train, uint inst, uint time, uint start_seed);