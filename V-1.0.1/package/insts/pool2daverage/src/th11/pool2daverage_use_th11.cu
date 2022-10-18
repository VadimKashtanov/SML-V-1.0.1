#include "package/insts/pool2daverage/head/pool2daverage.cuh"

__global__
void pool2daverage_use_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,
	uint time,
	uint total,
	uint istart, uint ystart,
	float * var)
{
	uint _Yx = threadIdx.x + blockIdx.x * blockDim.x;
	uint _Yy = threadIdx.y + blockIdx.y * blockDim.y;

	if (_Yx < Yx && _Yy < Yy) {
		float _sum = 0;

		for (uint _y=0; _y < Ypool; _y++) {
			for (uint _x=0; _x < Xpool; _x++) {
				_sum += var[time*total + istart + (_Yy*Ypool + _y)*Ax + (_Yx*Xpool + _x)];
			}
		}

		var[time*total + ystart + _Yy*Yx + _Yx] = _sum / (Xpool * Ypool);
	};
};