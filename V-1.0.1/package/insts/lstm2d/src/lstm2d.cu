#include "package/insts/lstm2d/head/lstm2d.cuh"

#ifndef logistic
#define logistic(x) 1 / (1 + exp(-x))
#endif

//			   0  1  2     3      4      5       6        7 
//Arguments = [Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate]

void lstm2d_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[2] == 0) raise(SIGINT);
	if (param[7] >100) raise(SIGINT);
};

void lstm2d_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5];

	uint total = mdl->total;

	uint _W = Bx * Ax;
	uint _U = Bx * Bx;
	uint _B = Bx * Ay;

	uint lineW = _W + _U + _B;

	float * var = cpu->var;
	float * weight = mdl->weight;

	float f0,f1,f2,g0;
	float xval,hval;
	float e,e_1,h;

	for (uint x=0; x < Bx; x++) {
		for (uint y=0; y < Ay; y++) {
			//	Compute f0,f1,f2
			f0 = 0; f1 = 0; f2 = 0; g0 = 0;

			//x@.W
			for (uint k=0; k < Ax; k++) {
				xval = var[total*time + istart + (y*Ax + k)];

				f0 += weight[wstart + 0*lineW + (k*Bx + y)] * xval;
				f1 += weight[wstart + 1*lineW + (k*Bx + y)] * xval;
				f2 += weight[wstart + 2*lineW + (k*Bx + y)] * xval;
				g0 += weight[wstart + 3*lineW + (k*Bx + y)] * xval;
			}

			//h[-1]@.U
			if (time > 0) {
				for (uint k=0; k < Bx; k++) {
					hval = var[total*(time-1) + ystart + (Bx*Ay) + (y*Bx + k)];	//h[-1] is stored after e (of size Bx*Ay)

					f0 += weight[wstart + 0*lineW + _W + (k*Bx + y)] * hval;
					f1 += weight[wstart + 1*lineW + _W + (k*Bx + y)] * hval;
					f2 += weight[wstart + 2*lineW + _W + (k*Bx + y)] * hval;
					g0 += weight[wstart + 3*lineW + _W + (k*Bx + y)] * hval;
				}
			}

			f0 += weight[wstart + 0*lineW + _W + _U + (y*Bx + x)];
			f1 += weight[wstart + 1*lineW + _W + _U + (y*Bx + x)];
			f2 += weight[wstart + 2*lineW + _W + _U + (y*Bx + x)];
			g0 += weight[wstart + 3*lineW + _W + _U + (y*Bx + x)];

			f0 = 1 / (1 + exp(-f0));
			f1 = 1 / (1 + exp(-f1));
			f2 = 1 / (1 + exp(-f2));
			g0 = tanh(g0);

			if (time > 0) e_1 = var[total*(time-1) + ystart + (y*Bx + x)];
			else e_1 = 0;

			e = f0 * e_1 + f1 * g0;
			h = f2 * e;

			var[total*time + ystart + (y*Bx + x)] = e;
			var[total*time + ystart + Bx*Ay + (y*Bx + x)] = h;
		}
	}
};

void lstm2d_use(Use_t * use, uint inst, uint time) {
	lstm2d_use_call_mode_th11(use, inst, time);
};

void lstm2d_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	lstm2d_forward_call_mode_th11(train, inst, time, start_seed);
};

void lstm2d_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	lstm2d_backward_call_mode_th11(train, inst, time, start_seed);
};