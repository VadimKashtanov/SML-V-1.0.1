#include "package/insts/dot2drecurent/head/dot2drecurent.cuh"

__global__
void dot2drecurent_use_th11(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	float _tmp;

	if (y < Ay && x < Bx) {

		_tmp = 0;

		for (uint i=0; i < Ax; i++) {
			_tmp += var[(time-At)*total + istart + y*Ax + i] * weight[wstart + i*Bx + y];
		}

		_tmp += weight[wstart + Bx*Ax + (y*Bx + x)];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

		if (activ == 0)	_tmp = 1 / (1 + exp(-_tmp));
		else if (activ == 1) _tmp = tanh(_tmp);
		else if (activ == 2) _tmp = exp(-_tmp*_tmp);
		else _tmp *= (_tmp > 0);

		var[time*total + ystart + (y*Bx + x)] = _tmp;
	}
};

__global__
void dot2drecurent_use_th11_NegativLine(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	float _tmp;

	if (y < Ay && x < Bx) {

		_tmp = weight[wstart + Bx*Ax + (y*Bx + x)];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

		if (activ == 0)	_tmp = 1 / (1 + exp(-_tmp));
		else if (activ == 1) _tmp = tanh(_tmp);
		else if (activ == 2) _tmp = exp(-_tmp*_tmp);
		else _tmp *= (_tmp > 0);

		var[time*total + ystart + (y*Bx + x)] = _tmp;
	}
};