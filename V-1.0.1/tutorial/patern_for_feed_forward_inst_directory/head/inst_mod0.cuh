#pragma once

#include "kernel/head/train.cuh"

//	----- USE -----

__global__
void inst_use_mod0(
	uint param0, uint parram1,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight);

//  --------- Forward -------- 

__global__
void inst_forward_mod0(
	uint param0, uint parram1,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets);

//  --------- Backward -------- 

__global__
void inst_backward_mod0(
	uint param0, uint parram1,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);