#include "package/insts/kconvl2d/head/kconvl2d.cuh"

void kconvl2d_check(uint * param) {
	uint Ax = param[0];
	uint Ay = param[1];
	uint Kx = param[2];
	uint Ky = param[3];
	uint n0 = param[4];
	uint n1 = param[5];
	uint strideX = param[6];
	uint strideY = param[7];
	//uint paddingX = param[8];
	//uint paddingY = param[9];
	uint activ = param[10];
	//uint input_start = param[11];
	//uint ystart = param[12];
	//uint wstart = param[13];
	//uint locdstart = param[14];
	uint drop_rate = param[15];

	assert(Kx % 2 != 0);
	assert(Ky % 2 != 0);
	assert(activ < 5);
	assert(strideX > 0);
	assert(strideY > 0);
	assert(Ax > 0);
	assert(Ay > 0);
	assert(n0 > 0);
	assert(n1 > 0);
	assert(Ax % strideX == 0);
	assert(Ay % strideY == 0);
	assert(drop_rate <= 100);
};

void kconvl2d_cpu(Cpu_t * cpu, uint inst, uint time) {
	//
	Mdl_t * mdl = cpu->mdl;
	uint total = mdl->total;

	//
	float * var = cpu->var;
	float * weight = mdl->weight;

	//
	uint * param = mdl->param[ inst ];
	uint Ax = param[0];
	uint Ay = param[1];
	uint Kx = param[2];
	uint Ky = param[3];
	uint n0 = param[4];
	uint n1 = param[5];
	uint strideX = param[6];
	uint strideY = param[7];
	uint paddingX = param[8];
	uint paddingY = param[9];
	uint activ = param[10];
	uint istart = param[11];
	uint ystart = param[12];
	uint wstart = param[13];
	//uint locdstart = param[14];
	//uint drop_rate = param[15];

	//
	uint ker_radiusX = (Kx-1)/2;
	uint ker_radiusY = (Ky-1)/2;

	uint Yx = (Ax - 2*paddingX) / strideX;
	uint Yy = (Ay - 2*paddingY) / strideY;

	float _sum;
	uint _pixelpos, _kernelpos;

	uint start_ker_x, end_ker_x, start_ker_y, end_ker_y;

	for (uint _n1=0; _n1 < n1; _n1++) {

		for (uint y=paddingY; y < Ay - paddingY; y += strideY) {
			for (uint x=paddingX; x < Ax - paddingX; x += strideX) {

				_sum = 0;

				start_ker_x = (x >= ker_radiusX) ? 0 : (ker_radiusX - x);	//max((ker_radius-x), 0)  it's kind of distance beetwin kernel border and image border
				start_ker_y = (y >= ker_radiusY) ? 0 : (ker_radiusY - y);

				end_ker_x = (x < (Ax - ker_radiusX)) ? Kx : (Kx - ((x+ker_radiusX) - Ax+1));
				end_ker_y = (y < (Ay - ker_radiusY)) ? Ky : (Ky - ((y+ker_radiusY) - Ay+1));

				for (uint _n0=0; _n0 < n0; _n0++) {
					for (uint ker_y=start_ker_y; ker_y < end_ker_y; ker_y++) {
						for (uint ker_x=start_ker_x; ker_x < end_ker_x; ker_x++) {
							_pixelpos = time*total + istart + _n0*Ax*Ay + (y+ker_y-ker_radiusY)*Ax + (x+ker_x-ker_radiusX);
							_kernelpos = wstart + _n1*Kx*Ky*n0 + _n0*Kx*Ky + (ker_y*Kx + ker_x);

							_sum += var[_pixelpos] * weight[_kernelpos];
						}
					}
				}

				_pixelpos = _n1*Yx*Yy + ((y-paddingY)/strideY)*Yx + ((x-paddingX)/strideX);
				_sum += weight[wstart + n1*n0*Kx*Ky + _pixelpos];

				if (activ == 0) _sum = 1 / (1 + exp(-_sum));
				else if (activ == 1) _sum = tanh(_sum);
				else if (activ == 2) _sum = exp(-_sum * _sum);
				else if (activ == 3) _sum = _sum * (_sum > 0);
				//else _tmp = tmp
				
				var[time*total + ystart + _pixelpos] = _sum;
			}
		}
	}
};

void kconvl2d_use(Use_t * use, uint inst, uint time) {
	//the only mode is th11
	kconvl2d_use_call_mode_th11(use, inst, time);
};

void kconvl2d_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	kconvl2d_forward_call_mode_th11(train, inst, time, start_seed);
};

void kconvl2d_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	kconvl2d_backward_call_mode_th11(train, inst, time, start_seed);
};