#include "package/insts/pool2dmax/head/pool2dmax.cuh"

__global__
void pool2dmax_forward_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,						
	uint time,
	uint total, uint locds,
	uint istart, uint ystart, uint locdstart,
	uint sets,
	float * var, float * locd)
{
	uint _Yx = threadIdx.x + blockIdx.x * blockDim.x;
	uint _Yy = threadIdx.y + blockIdx.y * blockDim.y;
	uint _set = blockIdx.z;

	if (_Yx < Yx && _Yy < Yy) {
		float _max = var[time*sets*total + _set*total + istart + (_Yy*Ypool + 0)*Ax + (_Yx*Xpool + 0)];
		
		uint index = 0;

		float _compare;

		for (uint _y=0; _y < Ypool; _y++) {
			for (uint _x=0; _x < Xpool; _x++) {
				_compare = var[time*sets*total + _set*total + istart + (_Yy*Ypool + _y)*Ax + (_Yx*Xpool + _x)];
				if (_compare > _max) {
					_max = _compare;
					index = _y*Xpool + _x;
				}
			}
		}

		var[time*sets*total + _set*total + ystart + _Yy*Yx + _Yx] = _max;
		locd[time*sets*locds + _set*locds + locdstart + _Yy*Yx + _Yx] = (float)index;
	};
}