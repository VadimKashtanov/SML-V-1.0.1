#include "package/insts/pool2daverage/head/pool2daverage.cuh"

__global__
void pool2daverage_backward_th1x1(
	uint Yx, uint Yy,
	uint Ax, uint Ay, uint Xpool, uint Ypool,
	uint time,
	uint total, uint locds,
	uint istart, uint ystart,
	uint sets,
	float * var, float * locd,
	float * grad)
{
	uint _Yx = threadIdx.x + blockIdx.x * blockDim.x;
	uint _Yy = threadIdx.y + blockIdx.y * blockDim.y;
	uint _set = blockIdx.z;

	if (_Yx < Yx && _Yy < Yy) {
		float dl_dpoolmax = grad[time*sets*total + _set*total + ystart + _Yy*Yx + _Yx] / (Xpool * Ypool);

		for (uint _y=0; _y < Ypool; _y++) {
			for (uint _x=0; _x < Xpool; _x++) {
				atomicAdd(&grad[time*sets*total + _set*total + istart + (_Yy*Ypool + _y)*Ax + (_Yx*Xpool + _x)], dl_dpoolmax);
			}
		}

		//	Si j'ajoute Stride, il faudra conserver le atomicAdd
		//	Sans le stride je peux en réalité mettre que += car chaque pixel de l'input n'est utilisé que dans un seul pool
	}
}