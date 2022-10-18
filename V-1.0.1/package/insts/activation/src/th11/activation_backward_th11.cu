#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void activation_backward_th11(
	uint _len,
	uint activ,
	uint time,
	uint istart, uint ystart, uint locdstart,
	uint total, uint locdsize,
	float * var, float * locd, float * grad,
	uint sets)
{
	uint i = threadIdx.x + blockIdx.x*blockDim.x;
	uint set = blockIdx.y;

	if (i < _len) {
		float dlds = grad[time*sets*total + set*total + ystart + i] * locd[time*sets*locdsize + set*locdsize + locdstart + i];

		var[time*sets*total + set*total + istart + i] += dlds;
	}
};