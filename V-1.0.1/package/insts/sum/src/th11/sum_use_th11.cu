#include "package/insts/sum/head/sum.cuh"

__global__
void sum_use_th11(
	uint size, uint items,
	uint time,
	uint total,
	uint istart, uint ystart,
	float * var)
{
	uint pos = threadIdx.x + blockIdx.x * blockDim.x;

	if (pos < size) {
		float _sum = 0;

		for (uint j = 0; j < items; j++) {
			_sum += var[time*total + istart + j*size + pos];
		}

		var[time*total + ystart + pos] = _sum;
	}
}