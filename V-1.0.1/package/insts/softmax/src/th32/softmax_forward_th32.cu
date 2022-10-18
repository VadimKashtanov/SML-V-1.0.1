#include "package/insts/softmax/head/softmax.cuh"

__global__
void softmax_forward_th32(
	uint len,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart,
	uint sets,
	float * var)
{
	uint pos = threadIdx.x;
	uint set = blockIdx.x;

	if (pos < len) {
		float exped = exp(var[time*sets*total + set*total + istart + pos]);
		__shared__ float sum;
		if (pos == 0) sum = 0;
		__syncthreads();
		atomicAdd(&sum, exped);
		__syncthreads();
		var[time*sets*total + set*total + ystart + pos] = exped / sum;
	}
};