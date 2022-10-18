#pragma once

#include "kernel/head/train.cuh"
 
#include "sum_th11.cuh"

//F(a,b,c..) = a + b + c

/*

size = 3
   [1, 2, 3]
+  [6, 1, 5]
+  [1, 1, 0]	items = 4
+  [9, 1, 0]
=  [17,5, 8]

*/

//=========================== Sizes ===============================

/*
	inputs = len
	vars = len
	weights = 0
	locds = 0
*/

void sum_check(uint * param);

//======================= Cpu_t forward ===========================

void sum_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Model Forward ===========================

void sum_use_call_mode_th11(Use_t * use, uint inst, uint time);

void sum_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void sum_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void sum_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void sum_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void sum_backward(Train_t * train, uint inst, uint time, uint start_seed);