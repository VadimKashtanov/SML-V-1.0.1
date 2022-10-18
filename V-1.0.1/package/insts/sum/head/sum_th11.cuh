#pragma once

#include "kernel/head/train.cuh"

//======================= Model Forward ===========================

__global__
void sum_use_th11(
	uint size, uint items,
	uint time,
	uint vsize,
	uint istart, uint ystart,
	float * var);

//======================== Train_t =======================

//-------------------------- forward ---------------------

__global__
void sum_forward_th11(
	uint size, uint items,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart,
	uint sets,
	float * var);

//-------------------------- backward ---------------------

__global__
void sum_backward_th11(
	uint size, uint items,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart,
	uint sets,
	float * var, float * grad);