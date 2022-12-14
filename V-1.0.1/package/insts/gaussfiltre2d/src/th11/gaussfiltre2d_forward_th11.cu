#include "package/insts/gaussfiltre2d/head/gaussfiltre2d.cuh"

__global__
void gaussfiltre2d_forward_th11(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	float _tmp;

	if (x < X && y < Y) {
		_tmp = var[time*sets*total + set*total + istart + (y*X+x)] + weight[wsize*set + wstart + x];
		var[time*sets*total + set*total + ystart + (y*X+x)] = exp(-pow(_tmp,2));
		locd[time*sets*lsize + set*lsize + lstart + (y*X+x)] = -2*_tmp*exp(-pow(_tmp,2));
	}
};