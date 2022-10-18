#include "package/insts/inst/head/inst.cuh"

__global__
void inst_use_mod0(
	uint param0, uint parram1,
	uint time,
	uint total,
	uint input_start, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x;

	if (_Yx < parram1) {
		
	}
};