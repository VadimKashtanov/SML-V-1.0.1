#include "package/insts/dot2drecurent/head/dot2drecurent.cuh"

__global__
void dot2drecurent_backward_th11(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	float dlds;
	uint Apos, Wpos;

	if (y < Ay && x < Bx) {
		dlds = grad[time*sets*total + set*total + ystart + (y*Bx + x)] * locd[time*locdsize*sets + set*locdsize + locdstart + (y*Bx + x)];

		for (uint i=0; i < Ax; i++) {
			Apos = (time-At)*total*sets + set*total + istart + y*Ax + i;
			Wpos = wsize*set + wstart + y + i*Bx;

			if (pseudo_randomf(Apos*seed) >= drop_rate) {
				atomicAdd(grad + Apos, weight[Wpos] * dlds);
				atomicAdd(meand + Wpos, var[Apos] * dlds);
			}
		}

		meand[wsize*set + wstart + Bx*Ax + (y*Bx + x)] += dlds;
	}
}


__global__
void dot2drecurent_backward_th11_NegativLine(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	float dlds;

	if (y < Ay && x < Bx) {
		dlds = grad[time*sets*total + set*total + ystart + (y*Bx + x)] * locd[time*locdsize*sets + set*locdsize + locdstart + (y*Bx + x)];

		meand[wsize*set + wstart + Bx*Ax + (y*Bx + x)] += dlds;
	}
}