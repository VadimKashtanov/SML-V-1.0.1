#pragma once

#include "kernel/head/train.cuh"

//	----- USE -----

__global__
void activation_use_th11(
	uint _len,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart,
	float * var);

//  --------- Forward -------- 

__global__
void activation_forward_th11(
	uint _len,
	uint activ,
	uint time,
	uint istart, uint ystart, uint lstart,
	uint total, uint locdsize,
	float * var, float * locd,
	uint sets);

//  --------- Backward -------- 

__global__
void activation_backward_th11(
	uint _len,
	uint activ,
	uint time,
	uint istart, uint ystart, uint lstart,
	uint total, uint lsize,
	float * var, float * locd, float * grad,
	uint sets);