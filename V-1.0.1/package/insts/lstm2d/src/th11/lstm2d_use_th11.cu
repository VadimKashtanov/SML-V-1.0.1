#include "package/insts/lstm2d/head/lstm2d.cuh"

__global__
void lstm2d_use_th11(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < Bx && y < Ay)
	{
		uint inp = total*time + istart;
		uint W = wstart;
		uint out = total*time + ystart;
		//uint locdpos = locds*time + locdstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		uint vpos, wpos;

		// f0,f1,f2 = logistic(x@W + h[-1]@U + B)
		// g0 	  = tanh 	(x@W + h[-1]@U + B)
		float f0=0,f1=0,f2=0,g0=0;

		float tmpt;

		// .W
		for (uint k=0; k < Ax; k++) {	//for all in INPUT
			//	Positions
			vpos = inp + (y*Ax + k);

			//
			wpos = k*Bx + y;

			tmpt = var[vpos];
			f0 += tmpt * weight[W + 0*lineW + wpos];
			f1 += tmpt * weight[W + 1*lineW + wpos];
			f2 += tmpt * weight[W + 2*lineW + wpos];
			g0 += tmpt * weight[W + 3*lineW + wpos];
		}

		// .U
		if (time > 0) {
			for (uint k=0; k < Bx; k++) {
				vpos = total*(time-1) + ystart + (Bx*Ay) + y*Bx + k;	///h[-1]
				wpos = _W + k*Bx + y;

				tmpt = var[vpos];
				f0 += tmpt * weight[W + 0*lineW + wpos];
				f1 += tmpt * weight[W + 1*lineW + wpos];
				f2 += tmpt * weight[W + 2*lineW + wpos];
				g0 += tmpt * weight[W + 3*lineW + wpos];
			}
		}

		// .B
		wpos = _W + _U + y*Bx + x;
		f0 += weight[W + 0*lineW + wpos];
		f1 += weight[W + 1*lineW + wpos];
		f2 += weight[W + 2*lineW + wpos];
		g0 += weight[W + 3*lineW + wpos];

		// activ(_sum)
		f0 = 1 / (1 + exp(-f0));
		f1 = 1 / (1 + exp(-f1));
		f2 = 1 / (1 + exp(-f2));
		g0 = tanh(g0);

		// e = f0 * e[-1] + f1 * g0
		// l - 1 have to be >= 0
		float e_1;
		if (time > 0) e_1 = var[total*(time-1) + ystart + y*Bx + x];
		else e_1 = 0;
		
		float e = f0*e_1 + f1*g0;
		float h = f2 * e;

		var[out + 0*Bx*Ay + y*Bx + x] = e;
		var[out + 1*Bx*Ay + y*Bx + x] = h;
	};
};