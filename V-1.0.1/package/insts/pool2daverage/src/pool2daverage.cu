#include "package/insts/pool2daverage/head/pool2daverage.cuh"

void pool2daverage_check(uint * param) {
	uint Ax = param[0];
	uint Ay = param[1];
	uint poolX = param[2];
	uint poolY = param[3];
	//uint istart = param[4];
	//uint ystart = param[5];
	//uint locdstart = param[6];

	assert(Ax % poolX == 0);
	assert(Ay % poolY == 0);
}

void pool2daverage_cpu(Cpu_t * cpu, uint inst, uint time) {
	//
	Mdl_t * mdl = cpu->mdl;
	uint total = mdl->total;

	//
	float * var = cpu->var;

	//
	uint * param = mdl->param[ inst ];
	uint Ax = param[0];
	uint Ay = param[1];
	uint poolX = param[2];
	uint poolY = param[3];
	uint istart = param[4];
	uint ystart = param[5];
	//uint locdstart = param[6];

	uint Yx = Ax / poolX;
	uint Yy = Ay / poolY;

	float _sum;

	for (uint y=0; y < Yy; y++) {
		for (uint x=0; x < Yx; x++) {
			_sum = 0;

			for (uint _y=0; _y < poolY; _y++) {
				for (uint _x=0; _x < poolX; _x++) {
					_sum += var[time*total + istart + (y*poolY + _y)*Ax + (x*poolX + _x)];
				}
			}
			var[time*total + ystart + y*Yx + x] = _sum / (poolX * poolY);
		}
	}
};

//============================== GPU ====================================

void pool2daverage_use(Use_t * use, uint inst, uint time) {
	//the only mode is th11
	pool2daverage_use_call_mode_th11(use, inst, time);
};

void pool2daverage_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	pool2daverage_forward_call_mode_th11(train, inst, time, start_seed);
};

void pool2daverage_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	pool2daverage_backward_call_mode_th11(train, inst, time, start_seed);
};