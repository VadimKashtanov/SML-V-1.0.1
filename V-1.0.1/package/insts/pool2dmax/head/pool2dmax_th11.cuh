#pragma once

#include "kernel/head/train.cuh"

//Ax, Ay, Xpool, Ypool

//	================== Use ==================

__global__
void pool2dmax_use_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,
	uint time,
	uint total,
	uint istart, uint ystart,
	float * var);

//========================		Train_t	  =========================

//----------------------------- forward ---------------------------

__global__
void pool2dmax_forward_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,						
	uint time,
	uint total, uint locds,
	uint istart, uint ystart, uint locdstart,
	uint sets,
	float * var, float * locd);

//----------------------------- backward ---------------------------

__global__
void pool2dmax_backward_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,
	uint time,
	uint total, uint locds,
	uint istart, uint ystart, uint locdstart,
	uint sets,
	float * var, float * locd,
	float * grad);