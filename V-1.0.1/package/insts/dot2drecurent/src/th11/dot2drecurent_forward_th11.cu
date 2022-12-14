#include "package/insts/dot2drecurent/head/dot2drecurent.cuh"

__global__
void dot2drecurent_forward_th11(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	float _tmp, _locd;

	uint Apos, Wpos;

	if (y < Ay && x < Bx) {
		_tmp = 0;

		for (uint i=0; i < Ax; i++) {
			Apos = (time-At)*total*sets + set*total + istart + y*Ax + i;
			Wpos = wsize*set + wstart + y + i*Bx;

			if (pseudo_randomf(Apos*seed) >= drop_rate)
				_tmp += var[Apos] * weight[Wpos];
		}

		_tmp += weight[wsize*set + wstart + Bx*Ax + (y*Bx + x)];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

		if (activ == 0)	{
			_tmp = 1 / (1 + exp(-_tmp));
			_locd = _tmp * (1 - _tmp);

		} else if (activ == 1) {
			_tmp = tanh(_tmp);
			_locd = 1 - _tmp*_tmp;

		} else if (activ == 2) {
			_locd = -2*_tmp;
			_tmp = exp(-_tmp*_tmp);
			_locd = _tmp * _locd;
		} else {
			_locd = (_tmp > 0);
			_tmp = _tmp * _locd;
		}

		var[time*sets*total + set*total + ystart + (y*Bx + x)] = _tmp;
		locd[time*locdsize*sets + set*locdsize + locdstart + (y*Bx + x)] = _locd;
	}
}

__global__
void dot2drecurent_forward_th11_NegativLine(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	float _tmp, _locd;

	if (y < Ay && x < Bx) {
		_tmp = weight[wsize*set + wstart + Bx*Ax + (y*Bx + x)];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

		if (activ == 0)	{
			_tmp = 1 / (1 + exp(-_tmp));
			_locd = _tmp * (1 - _tmp);

		} else if (activ == 1) {
			_tmp = tanh(_tmp);
			_locd = 1 - _tmp*_tmp;

		} else if (activ == 2) {
			_locd = -2*_tmp;
			_tmp = exp(-_tmp*_tmp);
			_locd = _tmp * _locd;
		} else {
			_locd = (_tmp > 0);
			_tmp = _tmp * _locd;
		}

		var[time*sets*total + set*total + ystart + (y*Bx + x)] = _tmp;
		locd[time*locdsize*sets + set*locdsize + locdstart + (y*Bx + x)] = _locd;
	}
}