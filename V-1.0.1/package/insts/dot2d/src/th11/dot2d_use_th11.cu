#include "package/insts/dot2d/head/dot2d.cuh"

__global__
void dot2d_use_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint vars,
	uint input_start, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint Yx = threadIdx.x + blockIdx.x*blockDim.x,	\
		 Yy = threadIdx.y + blockIdx.y*blockDim.y;	\

	uint Apos = time*vars + input_start + Yy*Ax;
	uint Bpos = wstart + Yx;

	float sum = 0;
	for (uint i=0; i < Ax; i++) {
		sum += var[Apos] * weight[Bpos];
		Apos++;
		Bpos += Bx; 
	}
	sum += weight[wstart + Bx*Ax + Yy*Bx + Yx];

	if (activ == 0) sum = 1 / (1 + exp(-sum));
	else if (activ == 1) sum = tanh(sum);
	else if (activ == 2) sum = exp(-pow(sum,2));
	else if (activ == 3) sum = sum*(sum >= 0);
	//else _tmp = tmp

	var[time*vars + ystart + Yy*Bx + Yx] = sum;
};