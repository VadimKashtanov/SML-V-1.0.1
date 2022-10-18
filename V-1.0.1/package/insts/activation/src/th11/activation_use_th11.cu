#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void activation_use_th11(
	uint _len,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart,
	float * var)
{
	uint i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < _len) {
		float value = var[time*total + istart + i];

		if (activ == 0) value = 1 / (1 + exp(-value));
		else if (activ == 1) activ = tanh(value);
		else if (activ == 2) activ = exp(-value*value);
		else if (activ == 3) activ *= (activ > 0);

		var[time*total + ystart + i] = value;
	}
};