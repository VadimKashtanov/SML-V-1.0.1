#include "package/insts/pool2dmax/head/pool2dmax.cuh"

void pool2dmax_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	//
	Mdl_t * mdl = use->mdl;
	uint total = mdl->total;
	//uint weights = mdl->weights;
	//uint locds = mdl->locds;

	//
	uint * param = mdl->param[ inst ];
	uint Ax = param[0];
	uint Ay = param[1];
	uint Xpool = param[2];
	uint Ypool = param[3];
	uint istart = param[4];
	uint ystart = param[5];
	//uint locdstart = param[6];

	uint Yx = Ax / Xpool;
	uint Yy = Ay / Ypool;

	pool2dmax_use_th1x1<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32)), dim3(32,32)>>>(
		Yx, Yy,
		Ax, Ay,
		Xpool, Ypool,
		time,
		total,
		istart, ystart,
		use->var_d);

	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
}

void pool2dmax_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	//
	Mdl_t * mdl = train->mdl;
	uint total = mdl->total;
	//uint weights = mdl->weights;
	uint locds = mdl->locds;

	//
	uint * param = mdl->param[ inst ];
	uint Ax = param[0];
	uint Ay = param[1];
	uint Xpool = param[2];
	uint Ypool = param[3];
	uint istart = param[4];
	uint ystart = param[5];
	uint locdstart = param[6];

	uint Yx = Ax / Xpool;
	uint Yy = Ay / Ypool;

	pool2dmax_forward_th1x1<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32), train->sets), dim3(32,32, 1)>>>(
		Yx, Yy,
		Ax, Ay,
		Xpool, Ypool,				
		time,
		total, locds,
		istart, ystart, locdstart,
		train->sets,
		train->_var, train->_locd);
	
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
}

void pool2dmax_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	//
	Mdl_t * mdl = train->mdl;
	uint total = mdl->total;
	//uint weights = mdl->weights;
	uint locds = mdl->locds;

	//
	uint * param = mdl->param[ inst ];
	uint Ax = param[0];
	uint Ay = param[1];
	uint Xpool = param[2];
	uint Ypool = param[3];
	uint istart = param[4];
	uint ystart = param[5];
	uint locdstart = param[6];

	uint Yx = Ax / Xpool;
	uint Yy = Ay / Ypool;

	pool2dmax_backward_th1x1<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32), train->sets), dim3(32,32, 1)>>>(
		Yx, Yy,
		Ax, Ay,
		Xpool, Ypool,
		time,
		total, locds,
		istart, ystart, locdstart,
		train->sets,
		train->_var, train->_locd,
		train->_grad);

	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
}