#include "package/insts/inst/head/inst.cuh"

__global__
void inst_backward_mod0(
	uint param0, uint parram1,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	/*	Kernel coordinates	*/
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (_Yx < parram1) {
		
	}
};