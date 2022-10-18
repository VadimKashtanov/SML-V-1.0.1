#include "package/insts/sum/head/sum.cuh"

__global__
void sum_backward_th11(
	uint size, uint items,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart,
	uint sets,
	float * var, float * grad)
{
	uint pos = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (pos < size) {
		float dlds = grad[time*sets*total + set*total + ystart + pos];

		for (uint j = 0; j < items; j++) {
			grad[time*sets*total + set*total + istart + j*size + pos] += dlds;
		}
	}
}