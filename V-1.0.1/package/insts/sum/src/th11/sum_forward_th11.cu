#include "package/insts/sum/head/sum.cuh"

__global__
void sum_forward_th11(
	uint size, uint items,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart,
	uint sets,
	float * var)
{
	uint pos = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (pos < size) {
		float _sum = 0;

		for (uint j = 0; j < items; j++) {
			_sum += var[time*sets*total + set*total + istart + j*size + pos];
		}

		var[time*sets*total + set*total + ystart + pos] = _sum;
	}
};