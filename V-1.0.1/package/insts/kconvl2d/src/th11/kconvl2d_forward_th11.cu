#include "package/insts/kconvl2d/head/kconvl2d.cuh"

/*
	Each kernel compute one pixel for each channels (n1 pixels)

	<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32), _n1), dim3(32,32, 1)>>>
*/

__global__
void kconvl2d_forward_th1x1(
	uint Yx, uint Yy,
	uint n0, uint n1, uint Ax, uint Ay,
	uint Kx, uint Ky,
	uint strideX, uint strideY,
	uint paddingX, uint paddingY,
	uint activ,						
	uint time,
	uint total, uint weights, uint locds,
	uint istart, uint wstart, uint ystart, uint lstart,
	uint seed, float drop_rate,
	uint _set, uint sets,
	float * var, float * weight, float * locd)
{
	uint _Yx = threadIdx.x + blockIdx.x * blockDim.x;
	uint _Yy = threadIdx.y + blockIdx.y * blockDim.y;
	uint _n1 = blockIdx.z;

	uint x = _Yx*strideX + paddingX;
	uint y = _Yy*strideY + paddingY;

	if (_Yx < Yx && _Yy < Yy) {
		uint ker_radiusX = (Kx-1)/2;
		uint ker_radiusY = (Ky-1)/2;

		uint _pixelpos, _kernelpos;

		float _sum = 0;

		uint start_ker_x = (x >= ker_radiusX) ? 0 : (ker_radiusX - x);	//max((ker_radius-x), 0)  it's kind of distance beetwin kernel border and image border
		uint start_ker_y = (y >= ker_radiusY) ? 0 : (ker_radiusY - y);

		uint end_ker_x = (x < (Ax - ker_radiusX)) ? Kx : (Kx - ((x+ker_radiusX) - Ax+1));
		uint end_ker_y = (y < (Ay - ker_radiusY)) ? Ky : (Ky - ((y+ker_radiusY) - Ay+1));

		for (uint _n0=0; _n0 < n0; _n0++) {
			for (uint ker_y=start_ker_y; ker_y < end_ker_y; ker_y++) {
				for (uint ker_x=start_ker_x; ker_x < end_ker_x; ker_x++) {
					_pixelpos = time*sets*total + _set*total + istart + _n0*Ax*Ay + (y+ker_y-ker_radiusY)*Ax + (x+ker_x-ker_radiusX);
					_kernelpos = _set*weights + wstart + _n1*Kx*Ky*n0 + _n0*Kx*Ky + (ker_y*Kx + ker_x);

					_sum += var[_pixelpos] * weight[_kernelpos];
				}
			}
		}

		_pixelpos = _n1*Yx*Yy + ((y-paddingY)/strideY)*Yx + ((x-paddingX)/strideX);
		_sum += weight[_set*weights + wstart + n1*n0*Kx*Ky + _pixelpos];

		float __locd;

		if (activ == 0) {
			_sum = 1 / (1 + exp(-_sum));
			__locd = _sum*(1 - _sum);	//f'(x) = f(x)(1 - f(x))
		} else if (activ == 1) {
			_sum = tanh(_sum);
			__locd = 1 - _sum*_sum;		//f'(x) = 1 - tanh(x)^2
		} else if (activ == 2) {
			__locd = _sum;
			_sum = exp(-pow(_sum,2));
			__locd = -2*__locd*_sum;	//f'(x) = -2x*e^(-x^2)
		} else if (activ == 3) {
			__locd = (_sum >= 0);
			_sum = _sum*__locd;
		} else if (activ == 4) {
			__locd = 1;
		}

		var[time*sets*total + _set*total + ystart + _pixelpos] = _sum;		//same assembler than putting it in if/else structure
		locd[time*sets*locds + _set*locds + lstart + _pixelpos] = __locd;
	}
};