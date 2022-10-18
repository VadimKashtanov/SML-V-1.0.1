#include "package/insts/pool2dmax/head/pool2dmax.cuh"

__global__
void pool2dmax_backward_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,
	uint time,
	uint total, uint locds,
	uint istart, uint ystart, uint locdstart,
	uint sets,
	float * var, float * locd,
	float * grad)
{
	uint _Yx = threadIdx.x + blockIdx.x * blockDim.x;
	uint _Yy = threadIdx.y + blockIdx.y * blockDim.y;
	uint _set = blockIdx.z;

	if (_Yx < Yx && _Yy < Yy) {
		float dl_dpoolmax = grad[time*sets*total + _set*total + ystart + _Yy*Yx + _Yx];

		uint locd_val = (uint)locd[time*sets*locds + _set*locds + locdstart + _Yy*Yx + _Yx];

		uint _x = locd_val % Xpool;			//index en x du maximum du block
		uint _y = (locd_val-_x) / Xpool;	//et en y

		grad[time*sets*total + _set*total + istart + (_Yy*Ypool + _y)*Ax + (_Yx*Xpool + _x)] += dl_dpoolmax;
	}
}