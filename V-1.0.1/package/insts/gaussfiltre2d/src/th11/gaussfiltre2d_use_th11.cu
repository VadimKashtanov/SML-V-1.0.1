#include "package/insts/gaussfiltre2d/head/gaussfiltre2d.cuh"

__global__
void gaussfiltre2d_use_th11(
	uint X, uint Y,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < X && y < Y) {
		var[time*total + ystart + y*X + x] = exp(-pow(var[time*total + istart + y*X + x] + weight[wstart + x],2));
	}
};