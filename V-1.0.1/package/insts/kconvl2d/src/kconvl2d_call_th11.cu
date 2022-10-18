#include "package/insts/kconvl2d/head/kconvl2d.cuh"

void kconvl2d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	//
	Mdl_t * mdl = use->mdl;
	uint total = mdl->total;
	uint weights = mdl->weights;
	//uint locds = mdl->locds;

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

	uint Yx = (Ax - 2*paddingX) / strideX;
	uint Yy = (Ay - 2*paddingY) / strideY;

	kconvl2d_use_th1x1<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32), n1), dim3(32,32, 1)>>>(
		Yx, Yy,
		n0, n1, Ax, Ay,
		Kx, Ky,
		strideX, strideY,
		paddingX, paddingY,
		activ,						
		time,
		total, weights,
		istart, wstart, ystart,
		use->var_d, use->weight_d);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
}

void kconvl2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	//
	Mdl_t * mdl = train->mdl;
	uint total = mdl->total;
	uint weights = mdl->weights;
	uint locds = mdl->locds;

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
	uint locdstart = param[14];
	uint drop_rate = param[15];

	uint Yx = (Ax - 2*paddingX) / strideX;
	uint Yy = (Ay - 2*paddingY) / strideY;

	for (uint _set=0; _set < train->sets; _set++) {
		kconvl2d_forward_th1x1<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32), n1), dim3(32,32, 1)>>>(
			Yx, Yy,
			n0, n1, Ax, Ay,
			Kx, Ky,
			strideX, strideY,
			paddingX, paddingY,
			activ,
			time,
			total, weights, locds,
			istart, wstart, ystart, locdstart,
			start_seed, (float)drop_rate / 100,
			_set, train->sets,
			train->_var, train->_weight, train->_locd);
		cudaDeviceSynchronize();
		SAFE_CUDA(cudaPeekAtLastError());
	}
}

void kconvl2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	//
	Mdl_t * mdl = train->mdl;
	uint total = mdl->total;
	uint weights = mdl->weights;
	uint locds = mdl->locds;

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
	uint locdstart = param[14];
	uint drop_rate = param[15];

	uint Yx = (Ax - 2*paddingX) / strideX;
	uint Yy = (Ay - 2*paddingY) / strideY;

	for (uint _set=0; _set < train->sets; _set++) {
		kconvl2d_backward_th1x1<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32), n1), dim3(32,32, 1)>>>(
			Yx, Yy,
			n0, n1, Ax, Ay,
			Kx, Ky,
			strideX, strideY,
			paddingX, paddingY,
			activ,						
			time,
			total, weights, locds,
			istart, wstart, ystart, locdstart,
			start_seed, (float)drop_rate / 100,
			_set, train->sets,
			train->_var, train->_weight, train->_locd,
			train->_grad, train->_meand);
		cudaDeviceSynchronize();
		SAFE_CUDA(cudaPeekAtLastError());
	}
}