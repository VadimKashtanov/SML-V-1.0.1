#pragma once

#include "kernel/head/train.cuh"

//	----- USE -----

__global__
void gaussfiltre2d_use_th11(
	uint X, uint Y,
	uint time,
	uint vars,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight);

//  --------- Forward -------- 

__global__
void gaussfiltre2d_forward_th11(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint sets);

//  --------- Backward -------- 

__global__
void gaussfiltre2d_backward_th11(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint sets);