#include "package/insts/gaussfiltre2d/head/gaussfiltre2d.cuh"

__global__
void gaussfiltre2d_backward_th11(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	float dlds;

	if (x < X && y < Y) {
		dlds = grad[time*sets*total + set*total + ystart + (y*X+x)] * locd[time*sets*lsize + set*lsize + lstart + (y*X+x)];

		grad[time*sets*total + set*total + istart + (y*X+x)] += dlds;
		atomicAdd(meand + wsize*set + wstart + x, dlds);
	}
};