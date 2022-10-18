#include "package/insts/pool2daverage/head/pool2daverage.cuh"

__global__
void pool2daverage_forward_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,						
	uint time,
	uint total, uint locds,
	uint istart, uint ystart,
	uint sets,
	float * var, float * locd)
{
	uint _Yx = threadIdx.x + blockIdx.x * blockDim.x;
	uint _Yy = threadIdx.y + blockIdx.y * blockDim.y;
	uint _set = blockIdx.z;

	if (_Yx < Yx && _Yy < Yy) {
		float _sum = 0;

		for (uint _y=0; _y < Ypool; _y++) {
			for (uint _x=0; _x < Xpool; _x++) {
				_sum += var[time*sets*total + _set*total + istart + (_Yy*Ypool + _y)*Ax + (_Yx*Xpool + _x)];
			}
		}

		var[time*sets*total + _set*total + ystart + _Yy*Yx + _Yx] = _sum / (Xpool * Ypool);
	};
}