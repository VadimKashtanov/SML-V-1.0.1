#include "package/insts/dot1d/head/dot1d.cuh"

__global__
void activation_forward_th11(
	uint _len,
	uint activ,
	uint time,
	uint istart, uint ystart, uint locdstart,
	uint total, uint locdsize,
	float * var, float * locd,
	uint sets)
{
	uint i = threadIdx.x + blockIdx.x*blockDim.x;
	uint set = blockIdx.y;

	if (i < _len) {
		float value = var[time*sets*total + set*total + istart + i];
		float __locd;

		if (activ == 0) {
			value = 1 / (1 + exp(-value));
			__locd = value*(1 - value);	//f'(x) = f(x)(1 - f(x))
		} else if (activ == 1) {
			value = tanh(value);
			__locd = 1 - value*value;	//f'(x) = 1 - tanh(x)^2
		} else if (activ == 2) {
			__locd = value;
			value = exp(-value*value);
			__locd = -2*__locd*value;	//f'(x) = -2x*e^(-x^2)
		} else  if (activ == 3) {
			__locd = (value > 0);
			value = value*__locd;
		} else if (activ == 4) {
			__locd = 1;
		}

		var[time*sets*total + set*total + ystart + i] = value;		//same assembler than putting it in if/else structure
		locd[time*sets*locdsize + set*locdsize + locdstart + i] = __locd;
	}
};
